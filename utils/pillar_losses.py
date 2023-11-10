import torch
from pytorch_metric_learning.miners.base_miner import BaseTupleMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss


class PillarMiner(BaseTupleMiner):
    def __init__(self, epsilon, posDistThr=0.1, posRotThr=0.2, negDistThr=0.2, negRotThr=1.0, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.posDistThr = posDistThr
        self.posRotThr = posRotThr
        self.negDistThr = negDistThr
        self.negRotThr = negRotThr

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        pos_labels = labels[:, :3]
        rot_labels = labels[:, 3:]
        ref_pos_labels = ref_labels[:, :3]
        ref_rot_labels = ref_labels[:, 3:]

        mat = self.distance(embeddings, ref_emb)
        pos_a1, pos_p, pos_a2, pos_n = self.get_all_pairs_indices(pos_labels, ref_pos_labels, self.posDistThr)
        rot_a1, rot_p, rot_a2, rot_n = self.get_all_pairs_indices(rot_labels, ref_rot_labels, self.posRotThr)

        if len(pos_a1) == 0 or len(pos_a2) == 0 or len(rot_a1) == 0 or len(rot_a2) == 0:
            empty = torch.tensor([], device=labels.device, dtype=torch.long)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[pos_a2, pos_n] = pos_ignore
        mat_neg_sorting[pos_a1, pos_p] = neg_ignore
        mat_pos_sorting[rot_a2, rot_n] = pos_ignore
        mat_neg_sorting[rot_a1, rot_p] = neg_ignore
        if embeddings is ref_emb:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            hard_pos_idx = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]

        return a1, p, a2, n

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output

    def get_default_distance(self):
        return CosineSimilarity()

    def get_matches_and_diffs(self, labels, ref_labels=None, diff_thresh=0):
        if ref_labels is None:
            ref_labels = labels
        labels1 = labels.unsqueeze(1)
        labels2 = ref_labels.unsqueeze(0)
        matches = (torch.norm(abs(labels1 - labels2), dim=2) <= diff_thresh).byte()
        diffs = matches ^ 1
        if ref_labels is labels:
            matches.fill_diagonal_(0)
        return matches, diffs

    def get_all_pairs_indices(self, labels, ref_labels=None, diff_thresh=0):
        """
        Given a tensor of labels, this will return 4 tensors.
        The first 2 tensors are the indices which form all positive pairs
        The second 2 tensors are the indices which form all negative pairs
        """
        matches, diffs = self.get_matches_and_diffs(labels, ref_labels, diff_thresh)
        a1_idx, p_idx = torch.where(matches)
        a2_idx, n_idx = torch.where(diffs)
        return a1_idx, p_idx, a2_idx, n_idx

class PillarLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.add_to_recordable_attributes(
            list_of_names=["alpha", "beta", "base"], is_stat=False
        )

    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)
        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return CosineSimilarity()


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x

def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if not torch.is_tensor(ref_labels):
            TypeError("if ref_emb is given, then ref_labels must also be given")
        if ref_labels is not None:
            ref_labels = to_device(ref_labels, ref_emb)
    else:
        ref_emb, ref_labels = embeddings, labels
    return ref_emb, ref_labels
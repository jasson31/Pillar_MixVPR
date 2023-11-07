import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.miners.base_miner import BaseTupleMiner


class PillarMiner(BaseTupleMiner):
    def __init__(self, posDistThr=100, posRotThr=0.2, negDistThr=200, negRotThr=1.0, **kwargs):
        super().__init__(**kwargs)
        self.posDistThr = posDistThr
        self.posRotThr = posRotThr
        self.negDistThr = negDistThr
        self.negRotThr = negRotThr

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        pos_labels = labels[:3]
        rot_labels = labels[3:]
        ref_pos_labels = ref_labels[:3]
        ref_rot_labels = ref_labels[3:]

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

        mat_pos_sorting[pos_a1, pos_n] = pos_ignore
        mat_pos_sorting[rot_a1, rot_n] = pos_ignore
        mat_neg_sorting[pos_a1, pos_p] = neg_ignore
        mat_neg_sorting[rot_a1, rot_n] = neg_ignore
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

    def get_default_distance(self):
        return CosineSimilarity()

    def get_matches_and_diffs(self, labels, ref_labels=None, diff_thresh=0):
        if ref_labels is None:
            ref_labels = labels
        labels1 = labels.unsqueeze(1)
        labels2 = ref_labels.unsqueeze(0)
        matches = (abs(labels1 - labels2) <= diff_thresh).byte()
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

import numpy as np
import csv
import pandas as pd

dataset_directory = '../../../Dataset'
dataset_name = 'PillarDataset_Seg'
train_directory = 'train_val/Pillar_train'
val_directory = 'train_val/Pillar_val'
#test_directory = 'test/Pillar_test'


trainDatabaseReader = list(csv.reader(open(f'{dataset_directory}/{dataset_name}/{train_directory}/database/postprocessed.csv', 'r')))[1:]
trainQueryReader = list(csv.reader(open(f'{dataset_directory}/{dataset_name}/{train_directory}/query/postprocessed.csv', 'r')))[1:]

valDatabaseReader = list(csv.reader(open(f'{dataset_directory}/{dataset_name}/{val_directory}/database/postprocessed.csv', 'r')))[1:]
valQueryReader = list(csv.reader(open(f'{dataset_directory}/{dataset_name}/{val_directory}/query/postprocessed.csv', 'r')))[1:]

#testDatabaseReader = csv.reader(open(f'{dataset_directory}/{dataset_name}/{test_directory}/database/postprocessed.csv', 'r'))
#testQueryReader = csv.reader(open(f'{dataset_directory}/{dataset_name}/{test_directory}/query/postprocessed.csv', 'r'))

#next(testDatabaseReader)
#next(testQueryReader)

dataframe_list = []
dataframe_list.append(['place_id', 'year', 'month', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'name'])
for i, line in enumerate(trainDatabaseReader):
    dataframe_list.append([i, 2023, 11, float(line[2]) / 1000, 1.3, float(line[3]) / 1000, line[8], line[9], line[10], line[11], f'./{dataset_name}/{train_directory}/{line[1]}.png'])

pd.DataFrame(dataframe_list).to_csv('PillarDataframe.csv', index=False, header=False)

posDistThr = 0.2
rotDistThr = 1.0
val_pIdx = []
val_qIdx = []
val_qImages = []
val_dbImages = []
for query_index, query_line in enumerate(valQueryReader):
    q_pair = []
    query_pose = np.array([float(query_line[2]) / 1000, 1.3, float(query_line[3]) / 1000, float(query_line[8]), float(query_line[9]), float(query_line[10]), float(query_line[11])])
    val_qImages.append(f'{val_directory}/query/images/{query_line[1]}.png')
    for db_index, db_line in enumerate(valDatabaseReader):
        db_pose = np.array([float(db_line[2]) / 1000, 1.3, float(db_line[3]) / 1000, float(db_line[8]), float(db_line[9]), float(db_line[10]), float(db_line[11])])

        if np.linalg.norm(query_pose[:3] - db_pose[:3]) < posDistThr and np.linalg.norm(query_pose[3:] - db_pose[3:]) < rotDistThr:
            q_pair.append(db_index)
        if query_index == 0:
            val_dbImages.append(f'{val_directory}/query/images/{query_line[1]}.png')

    if len(q_pair) != 0:
        val_qIdx.append(query_index)
        val_pIdx.append(q_pair)

    print(f'{query_index} / {len(valQueryReader)}')


np.save(f'{dataset_name}_val_pIdx', val_pIdx)
np.save(f'{dataset_name}_val_qIdx', val_qIdx)
np.save(f'{dataset_name}_val_qImages', val_qImages)
np.save(f'{dataset_name}_val_dbImages', val_dbImages)

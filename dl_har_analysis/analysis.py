import pandas as pd
import numpy as np
import os
import json


def run_analysis(log_dir, testing=False):
    train_results = pd.read_csv(os.path.join('../logs', log_dir, 'train_results.csv'), index_col=None)
    avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = [], [], [], []
    avg_v_loss, avg_v_acc, avg_v_fm, avg_v_fw = [], [], [], []

    # average analysis
    for i, row in train_results.iterrows():
        if i == 0:
            avg_t_loss = json.loads(row['t_loss'])
            avg_t_acc = json.loads(row['t_acc'])
            avg_t_fm = json.loads(row['t_fm'])
            avg_t_fw = json.loads(row['t_fw'])
            avg_v_loss = json.loads(row['v_loss'])
            avg_v_acc = json.loads(row['v_acc'])
            avg_v_fm = json.loads(row['v_fm'])
            avg_v_fw = json.loads(row['v_fw'])
        else:
            avg_t_loss = np.add(avg_t_loss, json.loads(row['t_loss']))
            avg_t_acc = np.add(avg_t_acc, json.loads(row['t_acc']))
            avg_t_fm = np.add(avg_t_fm, json.loads(row['t_fm']))
            avg_t_fw = np.add(avg_t_fw, json.loads(row['t_fw']))
            avg_v_loss = np.add(avg_v_loss, json.loads(row['v_loss']))
            avg_v_acc = np.add(avg_v_acc, json.loads(row['v_acc']))
            avg_v_fm = np.add(avg_v_fm, json.loads(row['v_fm']))
            avg_v_fw = np.add(avg_v_fw, json.loads(row['v_fw']))

    avg_t_loss /= len(train_results)
    avg_t_acc /= len(train_results)
    avg_t_fm /= len(train_results)
    avg_t_fw /= len(train_results)
    avg_v_loss /= len(train_results)
    avg_v_acc /= len(train_results)
    avg_v_fm /= len(train_results)
    avg_v_fw /= len(train_results)

    pd.DataFrame((avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw, avg_v_loss, avg_v_acc, avg_v_fm, avg_v_fw), index=None).to_csv(os.path.join('../logs', log_dir, 'temp.csv'), index=False, header=False)

    # seed analysis
    #for seed in

    # subject-wise analysis
    for sbj in np.unique(train_results['sbj']):
        if sbj == -1:
            continue
        else:
            sbj_data = train_results[train_results.sbj == sbj]
            print(sbj_data)


if __name__ == '__main__':
    run_analysis('20211205/174522')

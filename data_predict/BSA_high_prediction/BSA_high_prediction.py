import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from data_predict.CODE import *
from S import *


if __name__ == '__main__':

    cur_path = os.getcwd()
    file_path = os.path.join(cur_path, 'BSA_data.xlsx')
    data_BSA = pd.read_excel(file_path, header=1)

    data_BSA_high_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'high'),
    ['BSA(mg/ml)','dissolved oxygen(%)']].values
    data_BSA_high_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'high'),
    ['BSA(mg/ml)','dissolved oxygen(%)']].values

    loss_high_recovery_k_9, x_high_recovery_k_9, y_high_recovery_k_9 = prediction(data_BSA_high_recovery, '0_1',
                                                                                  data_BSA_high_enrichment, 9, 'Z',
                                                                                  5.78)
    loss_high_enrichment_k_9, x_high_enrichment_k_9, y_high_enrichment_k_9 = prediction(data_BSA_high_enrichment, '1_0',
                                                                                        data_BSA_high_recovery, 9, 'Z',
                                                                                        8.27)

    best_k_high = 8.23
    loss_high_recovery_best_k, x_high_recovery_best_k, y_high_recovery_best_k = prediction(data_BSA_high_recovery, '0_1',
                                                                                        data_BSA_high_enrichment, best_k_high,
                                                                                        'Z', 5.78)
    loss_high_enrichment_best_k, x_high_enrichment_best_k, y_high_enrichment_best_k = prediction(data_BSA_high_enrichment, '1_0',
                                                                                                 data_BSA_high_recovery,
                                                                                                 best_k_high,
                                                                                              'Z', 8.27)
    with open(os.path.join(cur_path, 'BSA_low_prediction_results.txt'), 'w') as f:
        f.write(f'rmse_high_recovery_k_9 is {loss_high_recovery_k_9}\n')
        f.write(f'rmse_high_enrichment_k_9 is {loss_high_enrichment_k_9}\n')
        f.write(f'loss_high_recovery_best_k is {loss_high_recovery_best_k}\n')
        f.write(f'loss_high_enrichment_best_k is {loss_high_enrichment_best_k}\n')

    index_low = ['x', 'y']
    low_recovery_k_9 = pd.DataFrame([x_high_recovery_k_9, y_high_recovery_k_9], index=index_low).T
    low_recovery_k_9.to_csv('high_recovery_k_9.csv', index=False, header=True)

    low_enrichment_k_9 = pd.DataFrame([x_high_enrichment_k_9, y_high_enrichment_k_9], index=index_low).T
    low_enrichment_k_9.to_csv('high_enrichment_k_9.csv', index=False, header=True)

    low_recovery_best_k = pd.DataFrame([x_high_recovery_best_k, y_high_recovery_best_k], index=index_low).T
    low_recovery_best_k.to_csv('high_recovery_best_k.csv', index=False, header=True)

    low_enrichment_best_k = pd.DataFrame([x_high_enrichment_best_k, y_high_enrichment_best_k], index=index_low).T
    low_enrichment_best_k.to_csv('high_enrichment_best_k.csv', index=False, header=True)

    fig_14, ax_14 = plt.subplots(1, 2, figsize=(16, 4))
    ax_14[0].plot(x_high_recovery_k_9, y_high_recovery_k_9, linestyle='--')
    ax_14[0].plot(x_high_recovery_best_k, y_high_recovery_best_k)
    ax_14[0].scatter(data_BSA_high_enrichment[:, 0], data_BSA_high_enrichment[:, 1], s=20, c='black')
    ax_14[0].scatter(data_BSA_high_recovery[:, 0], data_BSA_high_recovery[:, 1], s=20, c='red')
    ax_14[0].set(xlim=(-0.5, 20), ylim=(-10, 20))

    ax_14[1].plot(x_high_enrichment_k_9, y_high_enrichment_k_9, linestyle='--')
    ax_14[1].plot(x_high_enrichment_best_k, y_high_enrichment_best_k)
    ax_14[1].scatter(data_BSA_high_enrichment[:, 0], data_BSA_high_enrichment[:, 1], s=20, c='black')
    ax_14[1].scatter(data_BSA_high_recovery[:, 0], data_BSA_high_recovery[:, 1], s=20, c='red')
    ax_14[1].set(xlim=(-0.5, 20), ylim=(-10, 20))
    # plt.show()
    plt.savefig('BSA_high_prediction.png', dpi=300)



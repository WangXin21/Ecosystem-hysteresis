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

    data_BSA_intermediate_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'intermediate'),
    ['BSA(mg/ml)','dissolved oxygen(%)']].values
    data_BSA_intermediate_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'intermediate'),
    ['BSA(mg/ml)','dissolved oxygen(%)']].values

    loss_intermediate_recovery_k_9, x_intermediate_recovery_k_9, y_intermediate_recovery_k_9 \
        = prediction(data_BSA_intermediate_recovery, '0_1',data_BSA_intermediate_enrichment, 9, 'Z',
                     1.87726)
    loss_intermediate_enrichment_k_9, x_intermediate_enrichment_k_9, y_intermediate_enrichment_k_9 \
        = prediction(data_BSA_intermediate_enrichment, '1_0',data_BSA_intermediate_recovery, 9, 'Z',
                     4.621761666666666)

    best_k_intermediate = 9.17616146879915
    loss_intermediate_recovery_best_k, x_intermediate_recovery_best_k, y_intermediate_recovery_best_k \
        = prediction(data_BSA_intermediate_recovery, '0_1',data_BSA_intermediate_enrichment, best_k_intermediate,
                     'Z', 1.87726)
    loss_intermediate_enrichment_best_k, x_intermediate_enrichment_best_k, y_intermediate_enrichment_best_k \
        = prediction(data_BSA_intermediate_enrichment, '1_0', data_BSA_intermediate_recovery, best_k_intermediate,
                     'Z', 4.621761666666666)
    with open(os.path.join(cur_path, 'BSA_low_prediction_results.txt'), 'w') as f:
        f.write(f'rmse_intermediate_recovery_k_9 is {loss_intermediate_recovery_k_9}\n')
        f.write(f'rmse_intermediate_enrichment_k_9 is {loss_intermediate_enrichment_k_9}\n')
        f.write(f'loss_intermediate_recovery_best_k is {loss_intermediate_recovery_best_k}\n')
        f.write(f'loss_intermediate_enrichment_best_k is {loss_intermediate_enrichment_best_k}\n')

    index_low = ['x', 'y']
    intermediate_recovery_k_9 = pd.DataFrame([x_intermediate_recovery_k_9, y_intermediate_recovery_k_9], index=index_low).T
    intermediate_recovery_k_9.to_csv('intermediate_recovery_k_9.csv', index=False, header=True)

    intermediate_enrichment_k_9 = pd.DataFrame([x_intermediate_enrichment_k_9, y_intermediate_enrichment_k_9], index=index_low).T
    intermediate_enrichment_k_9.to_csv('intermediate_enrichment_k_9.csv', index=False, header=True)

    intermediate_recovery_best_k = pd.DataFrame([x_intermediate_recovery_best_k, y_intermediate_recovery_best_k], index=index_low).T
    intermediate_recovery_best_k.to_csv('intermediate_recovery_best_k.csv', index=False, header=True)

    intermediate_enrichment_best_k = pd.DataFrame([x_intermediate_enrichment_best_k, y_intermediate_enrichment_best_k], index=index_low).T
    intermediate_enrichment_best_k.to_csv('intermediate_enrichment_best_k.csv', index=False, header=True)

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    ax[0].plot(x_intermediate_recovery_k_9, y_intermediate_recovery_k_9, linestyle='--')
    ax[0].plot(x_intermediate_recovery_best_k, y_intermediate_recovery_best_k)
    ax[0].scatter(data_BSA_intermediate_enrichment[:, 0], data_BSA_intermediate_enrichment[:, 1], s=20, c='black')
    ax[0].scatter(data_BSA_intermediate_recovery[:, 0], data_BSA_intermediate_recovery[:, 1], s=20, c='red')
    ax[0].set(xlim=(-0.2, 8.5), ylim=(-2.5, 20))

    ax[1].plot(x_intermediate_enrichment_k_9, y_intermediate_enrichment_k_9, linestyle='--')
    ax[1].plot(x_intermediate_enrichment_best_k, y_intermediate_enrichment_best_k)
    ax[1].scatter(data_BSA_intermediate_enrichment[:, 0], data_BSA_intermediate_enrichment[:, 1], s=20, c='black')
    ax[1].scatter(data_BSA_intermediate_recovery[:, 0], data_BSA_intermediate_recovery[:, 1], s=20, c='red')
    ax[1].set(xlim=(-0.2, 8.5), ylim=(-2.5, 20))
    # plt.show()
    plt.savefig('BSA_intermediate_prediction.png', dpi=300)



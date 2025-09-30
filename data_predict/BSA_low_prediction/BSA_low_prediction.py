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
    data_14 = pd.read_excel(file_path, header=1)

    d14_low = data_14.loc[data_14['enrichment treatment'] == 'low', ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    d14_low_recovery = data_14.loc[(data_14['state'] == 'recovery') & (data_14['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    d14_low_enrichment = data_14.loc[(data_14['state'] == 'enrichment') & (data_14['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    loss_low_recovery_k_9, x_low_recovery_k_9, y_low_recovery_k_9 = prediction(d14_low_recovery, '0_1', d14_low_enrichment, 9, 'Z', 0.7, True)
    loss_low_enrichment_k_9, x_low_enrichment_k_9, y_low_enrichment_k_9 = prediction(d14_low_enrichment, '1_0', d14_low_recovery, 9, 'Z', 1.1, True)
    print(loss_low_recovery_k_9)
    print(loss_low_enrichment_k_9)

    best_k = 9.313150737984731
    loss_low_recovery_best_k, x_low_recovery_best_k, y_low_recovery_best_k = prediction(d14_low_recovery, '0_1', d14_low_enrichment, best_k, 'Z', 0.7, True)
    loss_low_enrichment_best_k, x_low_enrichment_best_k, y_low_enrichment_best_k = prediction(d14_low_enrichment, '1_0', d14_low_recovery, best_k, 'Z', 1.1, True)
    print(loss_low_recovery_best_k)
    print(loss_low_enrichment_best_k)

    index_low = ['x', 'y']
    low_recovery_k_9 = pd.DataFrame([x_low_recovery_k_9, y_low_recovery_k_9], index=index_low).T
    low_recovery_k_9.to_csv('low_recovery_k_9.csv', index=False, header=True)

    low_enrichment_k_9 = pd.DataFrame([x_low_enrichment_k_9, y_low_enrichment_k_9], index=index_low).T
    low_enrichment_k_9.to_csv('low_enrichment_k_9.csv', index=False, header=True)

    low_recovery_best_k = pd.DataFrame([x_low_recovery_best_k, y_low_recovery_best_k], index=index_low).T
    low_recovery_best_k.to_csv('low_recovery_best_k.csv', index=False, header=True)

    low_enrichment_best_k = pd.DataFrame([x_low_enrichment_best_k, y_low_enrichment_best_k], index=index_low).T
    low_enrichment_best_k.to_csv('low_enrichment_best_k.csv', index=False, header=True)

    fig_14, ax_14 = plt.subplots(1, 2, figsize=(16, 4))
    ax_14[0].plot(x_low_recovery_k_9, y_low_recovery_k_9, linestyle='--')
    ax_14[0].plot(x_low_recovery_best_k, y_low_recovery_best_k)
    ax_14[0].scatter(d14_low_enrichment[:, 0], d14_low_enrichment[:, 1], s=20, c='black')
    ax_14[0].scatter(d14_low_recovery[:, 0], d14_low_recovery[:, 1], s=20, c='red')
    ax_14[0].set(xlim=(0.0, 3.0), ylim=(-2.5, 25))

    ax_14[1].plot(x_low_enrichment_k_9, y_low_enrichment_k_9, linestyle='--')
    ax_14[1].plot(x_low_enrichment_best_k, y_low_enrichment_best_k)
    ax_14[1].scatter(d14_low_enrichment[:, 0], d14_low_enrichment[:, 1], s=20, c='black')
    ax_14[1].scatter(d14_low_recovery[:, 0], d14_low_recovery[:, 1], s=20, c='red')
    ax_14[1].set(xlim=(0.0, 3.0), ylim=(-2.5, 25))
    # plt.show()
    plt.savefig('BSA_low_prediction.png', dpi=300)



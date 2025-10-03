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
    data1_0 = pd.read_excel('Pdata.xlsx', usecols='A:B', nrows=5).values
    data0_1 = pd.read_excel('Pdata.xlsx', usecols='D:E').values
    best_k = 13.18871688538545
    loss_0_1_k_9, x_0_1_k_9, y_0_1_k_9 = prediction(data0_1, '0_1', data1_0, 9, 'Z', 0.07353154459753422,
                                                                               True)
    loss_1_0_k_9, x_1_0_k_9, y_1_0_k_9 = prediction(data1_0, '1_0', data0_1, 9, 'Z',
                                                                                     0.22791878172588775, True)
    loss_0_1_best_k, x_0_1_best_k, y_0_1_best_k = prediction(data0_1, '0_1', data1_0, best_k, 'Z', 0.07353154459753422,
                                                    True)
    loss_1_0_best_k, x_1_0_best_k, y_1_0_best_k = prediction(data1_0, '1_0', data0_1, best_k, 'Z',
                                                    0.22791878172588775, True)

    index_low = ['x', 'y']
    P_0_1_k_9 = pd.DataFrame([x_0_1_k_9, y_0_1_k_9], index=index_low).T
    P_0_1_k_9.to_csv('P_0_1_k_9.csv', index=False, header=True)

    P_1_0_k_9 = pd.DataFrame([x_1_0_k_9, y_1_0_k_9], index=index_low).T
    P_1_0_k_9.to_csv('P_1_0_k_9.csv', index=False, header=True)

    P_0_1_best_k = pd.DataFrame([x_0_1_best_k, y_0_1_best_k], index=index_low).T
    P_0_1_best_k.to_csv('P_0_1_best_k.csv', index=False, header=True)

    P_1_0_best_k = pd.DataFrame([x_1_0_best_k, y_1_0_best_k], index=index_low).T
    P_1_0_best_k.to_csv('P_1_0_best_k.csv', index=False, header=True)

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    ax[0].plot(x_0_1_k_9, y_0_1_k_9, linestyle='--')
    ax[0].plot(x_0_1_best_k, y_0_1_best_k)
    ax[0].scatter(data1_0[:, 0], data1_0[:, 1], s=20, c='black')
    ax[0].scatter(data0_1[:, 0], data0_1[:, 1], s=20, c='red')
    ax[0].set(xlim=(0, 0.3), ylim=(-0.05, 0.5))

    ax[1].plot(x_1_0_k_9, y_1_0_k_9, linestyle='--')
    ax[1].plot(x_1_0_best_k, y_1_0_best_k)
    ax[1].scatter(data1_0[:, 0], data1_0[:, 1], s=20, c='black')
    ax[1].scatter(data0_1[:, 0], data0_1[:, 1], s=20, c='red')
    ax[1].set(xlim=(0, 0.3), ylim=(-0.05, 0.5))
    # plt.show()
    plt.savefig('P_prediction.png', dpi=300)
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
    cessation = pd.read_excel('data.xlsx', usecols='G,K').values[5:28, :]
    continous = pd.read_excel('data.xlsx', usecols='B,F').values[5:26, :]

    best_k = 9.111163270906282
    loss_cessation_best_k, x_cessation_best_k, y_cessation_best_k = prediction(cessation, '1_0', continous, best_k, 'S', -0.65)
    loss_continus_best_k, x_continous_best_k, y_continous_best_k = prediction(continous, '0_1', cessation, best_k, 'S', -0.2)

    loss_cessation_k_9, x_cessation_k_9, y_cessation_k_9 = prediction(cessation, '1_0', continous, 9, 'S', -0.65)
    loss_continus_k_9, x_continous_k_9, y_continous_k_9 = prediction(continous, '0_1', cessation, 9, 'S', -0.2)

    print(loss_cessation_best_k)
    print(loss_continus_best_k)

    index_low = ['x', 'y']
    cessation_best_k = pd.DataFrame([x_cessation_best_k, y_cessation_best_k], index=index_low).T
    cessation_best_k.to_csv('cessation_best_k.csv', index=False, header=True)

    cessation_k_9 = pd.DataFrame([x_cessation_k_9, y_cessation_k_9], index=index_low).T
    cessation_k_9.to_csv('cessation_k_9.csv', index=False, header=True)

    continus_best_k = pd.DataFrame([x_continous_best_k, y_continous_best_k], index=index_low).T
    continus_best_k.to_csv('continus_best_k.csv', index=False, header=True)

    continus_k_9 = pd.DataFrame([x_continous_k_9, y_continous_k_9], index=index_low).T
    continus_k_9.to_csv('continus_k_9.csv', index=False, header=True)

    mpl.rcParams['figure.dpi'] = 100
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x_cessation_best_k, y_cessation_best_k)
    ax[0].plot(x_cessation_k_9, y_cessation_k_9, linestyle= '--')
    ax[0].scatter(cessation.T[0], cessation.T[1], c='r')
    ax[0].scatter(continous.T[0], continous.T[1])
    ax[0].set(xlim=(-2, 2))
    ax[1].plot(x_continous_best_k, y_continous_best_k)
    ax[1].plot(x_continous_k_9, y_continous_k_9, '--')
    ax[1].scatter(cessation.T[0], cessation.T[1], c='r')
    ax[1].scatter(continous.T[0], continous.T[1])
    ax[1].set(xlim=(-2, 2))
    # plt.show()
    plt.savefig('Soil_nitrate_prediction.png', dpi=300)


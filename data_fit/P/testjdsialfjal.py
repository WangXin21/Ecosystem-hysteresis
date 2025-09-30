import os
import pandas as pd
import matplotlib.pyplot as plt
from S import *
import matplotlib as mpl
from scipy.signal import find_peaks

if __name__ == '__main__':
    cur_path = os.getcwd()
    output_path = os.path.join(cur_path, 'Output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data1_0 = pd.read_excel('Pdata.xlsx', usecols='A:B', nrows=5).values
    data0_1 = pd.read_excel('Pdata.xlsx', usecols='D:E').values
    # index = (data0_1[:, 1]>0.200) & (data0_1[:, 1]<0.3100)
    # print(index)
    tipping_point = [0.06, 0.2]
    rmse, R_squared, k, x, y, pre_0_1, pre_1_0, att, att_derivative = fit_data(data0_1, data1_0, shape='Z',
                                                                               data_TPP_x=None, k=13.191721715296822)
    print(f'k is {k}, R_squared is {R_squared}, rmse is {rmse}')

    mpl.rcParams['figure.dpi'] = 100
    # plt.style.use('seaborn-dark')
    fig, ax = plt.subplots()
    ax.plot(x, y, c='blue')
    ax.scatter(data1_0[:, 0], data1_0[:, 1], s=20, c='red')
    ax.scatter(data0_1[:, 0], data0_1[:, 1], s=20, c='black')
    ax.set(xlim=(0, 0.3), ylim=(-0.05, 0.5))
    # plt.fill_betweenx(y, x_, x__, alpha=0.6)
    fig.supylabel('Fraction of lake surface covered by charophyte vegetation')
    fig.supxlabel('Total P')
    plt.show()
    # plt.savefig(os.path.join(output_path, 'P.png'), dpi=300)
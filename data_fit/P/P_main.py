import os
import pandas as pd
import matplotlib.pyplot as plt
from S import *
import matplotlib as mpl


if __name__ == '__main__':
    cur_path = os.getcwd()
    output_path = os.path.join(cur_path, 'Output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data1_0 = pd.read_excel('Pdata.xlsx', usecols='A:B', nrows=5).values
    data0_1 = pd.read_excel('Pdata.xlsx', usecols='D:E').values
    tipping_point = [0.06, 0.2]
    rmse, R_squared, k, data_TPP_x, x, y, pre_0_1, pre_1_0, att, att_derivative = fit_data(data0_1, data1_0, shape='Z')
    # rmse_, R_squared_, k_, x_, y_, pre_0_1_, pre_1_0_, att_, att_derivative_ = fit_data(data0_1, data1_0, shape='Z', data_TPP_x=tipping_point, k=10.67)
    # rmse__, R_squared__, k__, x__, y__, pre_0_1__, pre_1_0__, att__, att_derivative__ = fit_data(data0_1, data1_0, shape='Z', data_TPP_x=tipping_point, k=11.67)

    index = ['y', 'x', 'x_0_1', 'x_1_0', 'attraction_0', 'attraction_1', 'total_attraction', 'att_derivative_0', 'att_derivative_1', 'total_att_derivative']
    P = pd.DataFrame([y, x, pre_0_1, pre_1_0, att[0], att[1], att[0]+att[1], att_derivative[0], att_derivative[1], att_derivative[0]+att_derivative[1]], index=index).T
    P.to_csv(os.path.join(output_path, 'P.csv'), index=False)
    print(f'k is {k}, R_squared is {R_squared}, rmse is {rmse}, data TPP is {data_TPP_x}')

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
    # plt.show()
    plt.savefig(os.path.join(output_path, 'P.png'), dpi=300)



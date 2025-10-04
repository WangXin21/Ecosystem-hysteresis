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
    k_interval = [np.float64(11.802088663670961), np.float64(13.496375022078352)]
    rmse, R_squared, k, data_TPP_x, x, y, pre_0_1, pre_1_0, att, att_derivative = fit_data(data0_1, data1_0, shape='Z')
    tipping_point =  [np.float64(0.07), np.float64(0.23)]
    _, _, _, _, x_lower_bound, _, _, _, _, _ = fit_data(data0_1, data1_0, shape='Z', k=k_interval[0])
    _, _, _, _, x_upper_bound, _, _, _, _, _ = fit_data(data0_1, data1_0, shape='Z', k=k_interval[1])

    index = ['y', 'x', 'x_0_1', 'x_1_0', 'attraction_0', 'attraction_1', 'total_attraction', 'att_derivative_0',
             'att_derivative_1', 'total_att_derivative']
    P = pd.DataFrame([y, x, pre_0_1, pre_1_0, att[0], att[1], att[0]+att[1], att_derivative[0], att_derivative[1],
                      att_derivative[0]+att_derivative[1]], index=index).T
    P.to_csv(os.path.join(output_path, 'P.csv'), index=False)
    with open(os.path.join(output_path, 'params.txt'), 'w') as f:
        f.write(f'k is {round(k, 2)}, R_squared is {round(R_squared, 2)}, rmse is {round(rmse, 2)},'
                f' data TPP is {round(data_TPP_x[0], 2), round(data_TPP_x[1], 2)}')

    # mpl.rcParams['figure.dpi'] = 100
    # plt.style.use('seaborn-dark')
    fig, ax = plt.subplots()
    ax.plot(x, y, c='blue')
    ax.scatter(data1_0[:, 0], data1_0[:, 1], s=20, c='red')
    ax.scatter(data0_1[:, 0], data0_1[:, 1], s=20, c='black')
    ax.set(xlim=(0, 0.3), ylim=(-0.05, 0.5))
    plt.fill_betweenx(y, x, x_lower_bound, alpha=0.4, color='yellow')
    plt.fill_betweenx(y, x, x_upper_bound, alpha=0.4, color='yellow')
    fig.supylabel('Fraction of lake surface covered by charophyte vegetation')
    fig.supxlabel('Total P')
    # plt.show()
    plt.savefig(os.path.join(output_path, 'P.png'), dpi=300)



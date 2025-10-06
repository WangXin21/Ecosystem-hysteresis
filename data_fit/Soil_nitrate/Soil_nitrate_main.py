import numpy as np
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
    cessation = pd.read_excel('data.xlsx', usecols='G,K').values[5:28, :]
    continous = pd.read_excel('data.xlsx', usecols='B,F').values[5:26, :]
    k_interval = [np.float64(8.963725692019297), np.float64(10.242240244571983)]
    rmse, R_squared, k, data_TPP_x, x, y, pre_0_1, pre_1_0, att, att_derivative = fit_data(cessation, continous, shape='S', data_TPP_x=[-0.65, -0.2])
    _, _, _, _, x_lower_bound, _, _, _, _, _ = fit_data(cessation, continous, shape='S', k=k_interval[0], data_TPP_x=[-0.65, -0.2])
    _, _, _, _, x_upper_bound, _, _, _, _, _ = fit_data(cessation, continous, shape='S', k=k_interval[1], data_TPP_x=[-0.65, -0.2])
    index = ['y', 'x', 'x_0_1', 'x_1_0', 'attraction_0', 'attraction_1', 'total_attraction', 'att_derivative_0',
             'att_derivative_1', 'total_att_derivative']
    Soil_nitrate = pd.DataFrame([y, x, pre_0_1, pre_1_0, att[0], att[1], att[0]+att[1], att_derivative[0],
                                 att_derivative[1], att_derivative[0]+att_derivative[1]], index=index).T
    Soil_nitrate.to_csv(os.path.join(output_path, 'Soil_nitrate_new.csv'), index=False)
    with open(os.path.join(output_path, 'params.txt'), 'w') as f:
        f.write(f'k is {round(k, 2)}, R_squared is {round(R_squared, 2)}, rmse is {round(rmse, 2)},'
                f' data TPP is {round(data_TPP_x[0], 2), round(data_TPP_x[1], 2)}')
    index_confidence = ['y', 'x_lower_bound', 'x_upper_bound']
    confidence_interval = pd.DataFrame([y, x_lower_bound, x_upper_bound],
                                       index=index_confidence).T
    confidence_interval.to_csv(os.path.join(output_path, 'Soil_nitrate_confidence_interval.csv'), header=True, index=False)

    mpl.rcParams['figure.dpi'] = 100
    # plt.style.use('seaborn-dark')
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.fill_betweenx(y, x_lower_bound, x_upper_bound, alpha=0.4, color='yellow')
    ax.scatter(cessation[:, 0], cessation[:, 1], s=20, c='blue')
    ax.scatter(continous[:, 0], continous[:, 1], s=20, c='red')
    plt.xlim(-2, 2)
    # plt.show()
    plt.savefig(os.path.join(output_path, 'Soil_nitrate.png'), dpi=300)





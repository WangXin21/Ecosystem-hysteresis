import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from S import *


def plot_figure(x, y, data0_1, data1_0):
    mpl.rcParams['figure.dpi'] = 100
    plt.style.use('seaborn-dark')
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.scatter(data1_0[:, 0], data1_0[:, 1], s=20, c='red')
    ax.scatter(data0_1[:, 0], data0_1[:, 1], s=20, c='blue')
    # ax.set(xlim=(0, 300), ylim=(0, 3))
    plt.show()


if __name__ == '__main__':
    cur_path = os.getcwd()
    output_path = os.path.join(cur_path, 'Output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_BSA = pd.read_excel('BSA_data.xlsx', header=1)

    data_BSA_low_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_low_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_intermediate_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'intermediate'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_intermediate_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'intermediate'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_high_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'high'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_high_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'high'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    rmse_low, R_squared_low, k_low, data_low_TPP_x, x_low, y_low, pre_x_0_1_low, pre_x_1_0_low, attraction_low, att_derivative_low = fit_data(data_BSA_low_recovery, data_BSA_low_enrichment, shape='Z')
    rmse_intermediate, R_squared_intermediate, k_intermediate, data_intermediate_TPP_x, x_intermediate, y_intermediate, pre_x_0_1_intermediate, pre_x_1_0_intermediate, attraction_intermediate, att_derivative_intermediate = fit_data(data_BSA_intermediate_recovery, data_BSA_intermediate_enrichment, shape='Z')
    rmse_high, R_squared_high, k_high, data_high_TPP_x, x_high, y_high, pre_x_0_1_high, pre_x_1_0_high, attraction_high, att_derivative_high = fit_data(data_BSA_high_recovery, data_BSA_high_enrichment, shape='Z')

    # save params
    with open(os.path.join(output_path, 'params.txt'), 'w') as f:
        f.write(f'k_low is {k_low}, R_squared is {R_squared_low}, rmse is {rmse_low}, data_low_TPP_x is {data_low_TPP_x}\n')
        f.write(f'k_intermediate is {k_intermediate}, R_squared is {R_squared_intermediate}, rmse is {rmse_intermediate}, data_intermediate_TPP_x is {data_intermediate_TPP_x}\n')
        f.write(f'k_high is {k_high}, R_squared is {R_squared_high}, rmse is {rmse_high}, data_high_TPP_x is {data_high_TPP_x}')

    # save fit curve
    index_low = ['y', 'x', 'x_0_1', 'x_1_0', 'attraction_0', 'attraction_1', 'total_attraction', 'att_derivative_0', 'att_derivative_1', 'total_att_derivative']
    low = pd.DataFrame([y_low, x_low, pre_x_0_1_low, pre_x_1_0_low, attraction_low[0], attraction_low[1], attraction_low[0]+attraction_low[1], att_derivative_low[0], att_derivative_low[1], att_derivative_low[0]+att_derivative_low[1]], index=index_low).T
    intermediate = pd.DataFrame([y_intermediate, x_intermediate, pre_x_0_1_intermediate, pre_x_1_0_intermediate, attraction_intermediate[0], attraction_intermediate[1], attraction_intermediate[0]+attraction_intermediate[1], att_derivative_intermediate[0], att_derivative_intermediate[1], att_derivative_intermediate[0]+att_derivative_intermediate[1]], index=index_low).T
    high = pd.DataFrame([y_high, x_high, pre_x_0_1_high, pre_x_1_0_high, attraction_high[0], attraction_high[1], attraction_high[0]+attraction_high[1], att_derivative_high[0], att_derivative_high[1], att_derivative_high[0]+att_derivative_high[1]], index=index_low).T
    low.to_csv(os.path.join(output_path, 'BSA_low.csv'), header=True, index=False)
    intermediate.to_csv(os.path.join(output_path, 'BSA_intermediate.csv'), header=True, index=False)
    high.to_csv(os.path.join(output_path, 'BSA_high.csv'), header=True, index=False)

    k_low_interval = [9.07649756536342, 9.782811315799238]
    k_intermediate_interval = [8.877061428162142, 9.748145610256376]
    k_high_interval = [8.14052351570614, 8.690841591199078]

    k_low_lower_bound = k_low_interval[0]
    k_low_upper_bound = k_low_interval[1]
    k_intermediate_lower_bound = k_intermediate_interval[0]
    k_intermediate_upper_bound = k_intermediate_interval[1]
    k_high_lower_bound = k_high_interval[0]
    k_high_upper_bound = k_high_interval[1]
    _, _, _, _, x_low_lower_bound, y_low_lower_bound, _, _, _, _ = fit_data(data_BSA_low_recovery, data_BSA_low_enrichment, shape='Z', k=k_low_lower_bound)
    _, _, _, _, x_low_upper_bound, y_low_upper_bound, _, _, _, _ = fit_data(data_BSA_low_recovery, data_BSA_low_enrichment, shape='Z', k=k_low_upper_bound)
    _, _, _, _, x_intermediate_lower_bound, y_intermediate_lower_bound, _, _, _, _ = fit_data(data_BSA_intermediate_recovery, data_BSA_intermediate_enrichment, shape='Z', k=k_intermediate_lower_bound)
    _, _, _, _, x_intermediate_upper_bound, y_intermediate_upper_bound, _, _, _, _ = fit_data(data_BSA_intermediate_recovery, data_BSA_intermediate_enrichment, shape='Z', k=k_intermediate_upper_bound)
    _, _, _, _, x_high_lower_bound, _, _, _, _, _ = fit_data(data_BSA_high_recovery, data_BSA_high_enrichment, shape='Z', k=k_high_lower_bound)
    _, _, _, _, x_high_upper_bound, _, _, _, _, _ = fit_data(data_BSA_high_recovery, data_BSA_high_enrichment, shape='Z', k=k_high_upper_bound)
    # plot fig
    fig_14, ax_14 = plt.subplots(1, 3, figsize=(8, 8))
    ax_14[0].plot(x_low, y_low, color='blue')
    ax_14[0].fill_betweenx(y_low, x_low, x_low_lower_bound, alpha=0.4, color='yellow')
    ax_14[0].fill_betweenx(y_low, x_low_upper_bound, x_low, alpha=0.4, color='yellow')
    ax_14[0].scatter(data_BSA_low_enrichment[:, 0], data_BSA_low_enrichment[:, 1], s=20, c='red')
    ax_14[0].scatter(data_BSA_low_recovery[:, 0], data_BSA_low_recovery[:, 1], s=20, c='black')
    ax_14[0].set(xlim=(0.4, 2.6), ylim=(-10, 20))

    ax_14[1].plot(x_intermediate, y_intermediate, color='blue')
    ax_14[1].fill_betweenx(y_intermediate, x_intermediate, x_intermediate_lower_bound, alpha=0.6, color='yellow')
    ax_14[1].fill_betweenx(y_intermediate, x_intermediate, x_intermediate_upper_bound, alpha=0.6, color='yellow')
    ax_14[1].scatter(data_BSA_intermediate_enrichment[:, 0], data_BSA_intermediate_enrichment[:, 1], s=20, c='red')
    ax_14[1].scatter(data_BSA_intermediate_recovery[:, 0], data_BSA_intermediate_recovery[:, 1], s=20, c='black')
    ax_14[1].set(xlim=(-0.2, 8.5), ylim=(-10, 20))

    ax_14[2].plot(x_high, y_high)
    ax_14[2].fill_betweenx(y_high, x_high, x_high_lower_bound, alpha=0.4, color='yellow')
    ax_14[2].fill_betweenx(y_high, x_high, x_high_upper_bound, alpha=0.4, color='yellow')
    ax_14[2].scatter(data_BSA_high_enrichment[:, 0], data_BSA_high_enrichment[:, 1], s=20, c='red')
    ax_14[2].scatter(data_BSA_high_recovery[:, 0], data_BSA_high_recovery[:, 1], s=20, c='black')
    ax_14[2].set(xlim=(-0.5, 20), ylim=(-10, 20))
    plt.savefig(os.path.join(output_path, 'BSA.png'), dpi=300)
    # plt.show()




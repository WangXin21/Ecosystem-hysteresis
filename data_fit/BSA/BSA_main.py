import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from S import *


if __name__ == '__main__':
    cur_path = os.getcwd()
    output_path = os.path.join(cur_path, 'Output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_BSA = pd.read_excel('BSA_data.xlsx', header=1)

    data_BSA_low_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'low'),
    ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_low_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'low'),
    ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_intermediate_recovery = data_BSA.loc[
        (data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'intermediate'),
        ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_intermediate_enrichment = data_BSA.loc[
        (data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'intermediate'),
        ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_high_recovery = data_BSA.loc[
        (data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'high'),
        ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_high_enrichment = data_BSA.loc[
        (data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'high'),
        ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    (rmse_low, R_squared_low, k_low, data_low_TPP_x, x_low, y_low, pre_x_0_1_low, pre_x_1_0_low, attraction_low,
     att_derivative_low) \
        = fit_data(data_BSA_low_recovery, data_BSA_low_enrichment, shape='Z')
    (rmse_intermediate, R_squared_intermediate, k_intermediate, data_intermediate_TPP_x, x_intermediate, y_intermediate,
     pre_x_0_1_intermediate, pre_x_1_0_intermediate, attraction_intermediate, att_derivative_intermediate) \
        = fit_data(data_BSA_intermediate_recovery, data_BSA_intermediate_enrichment, shape='Z')
    (rmse_high, R_squared_high, k_high, data_high_TPP_x, x_high, y_high, pre_x_0_1_high, pre_x_1_0_high, attraction_high,
     att_derivative_high) \
        = fit_data(data_BSA_high_recovery, data_BSA_high_enrichment, shape='Z')

    # save params
    with open(os.path.join(output_path, 'params.txt'), 'w') as f:
        f.write(f'k_low is {round(k_low, 2)}, R_squared is {round(R_squared_low, 2)}, rmse is {round(rmse_low, 2)}, '
                f'data_low_TPP_x is {round(data_low_TPP_x[0], 2), round(data_low_TPP_x[1], 2)}\n')
        f.write(f'k_intermediate is {round(k_intermediate, 2)}, R_squared is {round(R_squared_intermediate,2)}, '
                f'rmse is {round(rmse_intermediate, 2)}, data_intermediate_TPP_x is {round(data_intermediate_TPP_x[0], 2),
                round(data_intermediate_TPP_x[1], 2)}\n')
        f.write(f'k_high is {round(k_high, 2)}, R_squared is {round(R_squared_high, 2)}, rmse is {round(rmse_high,2)},'
                f' data_high_TPP_x is {round(data_high_TPP_x[0], 2), round(data_high_TPP_x[1], 2)}')

    # save fit curve
    index_low = ['y', 'x', 'x_0_1', 'x_1_0', 'attraction_0', 'attraction_1', 'total_attraction', 'att_derivative_0',
                 'att_derivative_1', 'total_att_derivative']
    low = pd.DataFrame([y_low, x_low, pre_x_0_1_low, pre_x_1_0_low, attraction_low[0], attraction_low[1],
                        attraction_low[0]+attraction_low[1], att_derivative_low[0], att_derivative_low[1],
                        att_derivative_low[0]+att_derivative_low[1]], index=index_low).T
    intermediate = pd.DataFrame([y_intermediate, x_intermediate, pre_x_0_1_intermediate, pre_x_1_0_intermediate,
                                 attraction_intermediate[0], attraction_intermediate[1],
                                 attraction_intermediate[0]+attraction_intermediate[1],
                                 att_derivative_intermediate[0], att_derivative_intermediate[1],
                                 att_derivative_intermediate[0]+att_derivative_intermediate[1]], index=index_low).T
    high = pd.DataFrame([y_high, x_high, pre_x_0_1_high, pre_x_1_0_high, attraction_high[0], attraction_high[1],
                         attraction_high[0]+attraction_high[1], att_derivative_high[0], att_derivative_high[1],
                         att_derivative_high[0]+att_derivative_high[1]], index=index_low).T
    low.to_csv(os.path.join(output_path, 'BSA_low.csv'), header=True, index=False)
    intermediate.to_csv(os.path.join(output_path, 'BSA_intermediate.csv'), header=True, index=False)
    high.to_csv(os.path.join(output_path, 'BSA_high.csv'), header=True, index=False)

    k_low_interval = [9.08, 9.78]
    k_intermediate_interval = [8.88, 9.75]
    k_high_interval = [8.14, 8.69]

    k_low_lower_bound = k_low_interval[0]
    k_low_upper_bound = k_low_interval[1]
    k_intermediate_lower_bound = k_intermediate_interval[0]
    k_intermediate_upper_bound = k_intermediate_interval[1]
    k_high_lower_bound = k_high_interval[0]
    k_high_upper_bound = k_high_interval[1]
    _, _, _, _, x_low_lower_bound, y_low_lower_bound, _, _, _, _ = fit_data(data_BSA_low_recovery, data_BSA_low_enrichment,
                                                                            shape='Z', k=k_low_lower_bound)
    _, _, _, _, x_low_upper_bound, y_low_upper_bound, _, _, _, _ = fit_data(data_BSA_low_recovery, data_BSA_low_enrichment,
                                                                            shape='Z', k=k_low_upper_bound)
    _, _, _, _, x_intermediate_lower_bound, y_intermediate_lower_bound, _, _, _, _ = fit_data(data_BSA_intermediate_recovery,
                                                                                              data_BSA_intermediate_enrichment,
                                                                                              shape='Z',
                                                                                              k=k_intermediate_lower_bound)
    _, _, _, _, x_intermediate_upper_bound, y_intermediate_upper_bound, _, _, _, _ = fit_data(data_BSA_intermediate_recovery,
                                                                                              data_BSA_intermediate_enrichment,
                                                                                              shape='Z',
                                                                                              k=k_intermediate_upper_bound)
    _, _, _, _, x_high_lower_bound, _, _, _, _, _ = fit_data(data_BSA_high_recovery, data_BSA_high_enrichment, shape='Z',
                                                             k=k_high_lower_bound)
    _, _, _, _, x_high_upper_bound, _, _, _, _, _ = fit_data(data_BSA_high_recovery, data_BSA_high_enrichment, shape='Z',
                                                             k=k_high_upper_bound)

    index_confidence = ['y', 'x_lower_bound', 'x_upper_bound']
    low_BSA_confidence = pd.DataFrame([y_low_lower_bound, x_low_lower_bound, x_low_upper_bound],
                                      index=index_confidence).T
    low_BSA_confidence.to_csv(os.path.join(output_path, 'BSA_low_confidence_interval.csv'), header=True, index=False)
    intermediate_BSA_confidence = pd.DataFrame([y_intermediate_lower_bound, x_intermediate_lower_bound,
                                                x_intermediate_upper_bound], index=index_confidence).T
    intermediate_BSA_confidence.to_csv(os.path.join(output_path, 'BSA_intermediate_confidence_interval.csv'), header=True, index=False)
    high_BSA_confidence = pd.DataFrame([y_high, x_high_lower_bound, x_high_upper_bound], index=index_confidence).T
    high_BSA_confidence.to_csv(os.path.join(output_path, 'BSA_high_confidence_interval.csv'), header=True, index=False)
    # plot fig
    fig, ax = plt.subplots(1, 3, figsize=(8, 8))
    ax[0].plot(x_low, y_low, color='blue')
    ax[0].fill_betweenx(y_low, x_low, x_low_lower_bound, alpha=0.4, color='yellow')
    ax[0].fill_betweenx(y_low, x_low_upper_bound, x_low, alpha=0.4, color='yellow')
    ax[0].scatter(data_BSA_low_enrichment[:, 0], data_BSA_low_enrichment[:, 1], s=20, c='red')
    ax[0].scatter(data_BSA_low_recovery[:, 0], data_BSA_low_recovery[:, 1], s=20, c='black')
    ax[0].set(xlim=(0.4, 2.6), ylim=(-10, 20))

    ax[1].plot(x_intermediate, y_intermediate, color='blue')
    ax[1].fill_betweenx(y_intermediate, x_intermediate, x_intermediate_lower_bound, alpha=0.6, color='yellow')
    ax[1].fill_betweenx(y_intermediate, x_intermediate, x_intermediate_upper_bound, alpha=0.6, color='yellow')
    ax[1].scatter(data_BSA_intermediate_enrichment[:, 0], data_BSA_intermediate_enrichment[:, 1], s=20, c='red')
    ax[1].scatter(data_BSA_intermediate_recovery[:, 0], data_BSA_intermediate_recovery[:, 1], s=20, c='black')
    ax[1].set(xlim=(-0.2, 8.5), ylim=(-10, 20))

    ax[2].plot(x_high, y_high)
    ax[2].fill_betweenx(y_high, x_high, x_high_lower_bound, alpha=0.4, color='yellow')
    ax[2].fill_betweenx(y_high, x_high, x_high_upper_bound, alpha=0.4, color='yellow')
    ax[2].scatter(data_BSA_high_enrichment[:, 0], data_BSA_high_enrichment[:, 1], s=20, c='red')
    ax[2].scatter(data_BSA_high_recovery[:, 0], data_BSA_high_recovery[:, 1], s=20, c='black')
    ax[2].set(xlim=(-0.5, 20), ylim=(-10, 20))
    plt.savefig(os.path.join(output_path, 'BSA.png'), dpi=300)
    # plt.show()




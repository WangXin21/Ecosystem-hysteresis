import pandas as pd
import os
from data_fit.bootstrap import *

if __name__ == '__main__':
    cur_path = os.getcwd()


    # BSA
    #
    BSA_file_path = os.path.join(cur_path, 'BSA', 'BSA_data.xlsx')
    data_BSA = pd.read_excel(BSA_file_path, header=1)

    data_BSA_low_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_low_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_intermediate_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'intermediate'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_intermediate_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'intermediate'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_high_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'high'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_high_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'high'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    BSA_low_k_interval, BSA_low_rmse_interval, BSA_low_R_squared_interval = bootstrap(data_BSA_low_recovery, data_BSA_low_enrichment, 'Z', n_bootstraps=10000, confidence_level=95, data_TPP_x=[0.55398488, 1.206557857142857])
    #
    BSA_intermediate_k_interval, BSA_intermediate_rmse_interval, BSA_intermediate_R_squared_interval = bootstrap(data_BSA_intermediate_enrichment, data_BSA_intermediate_recovery, 'Z', [1.87726, 4.621761666666666], n_bootstraps=10000, confidence_level=95)
    #
    BSA_high_k_interval, BSA_high_rmse_interval, BSA_high_R_squared_interval = bootstrap(data_BSA_high_enrichment, data_BSA_high_recovery, 'Z', [5.781249999999999, 8.269092783333333], n_bootstraps=10000, confidence_level=95)


    # P

    P_file_path = os.path.join(cur_path, 'P', 'Pdata.xlsx')
    data1_0 = pd.read_excel(P_file_path, usecols='A:B', nrows=5).values
    data0_1 = pd.read_excel(P_file_path, usecols='D:E').values
    P_k_interval, P_rmse_interval, P_R_squared_interval = bootstrap(data0_1, data1_0, 'Z', [np.float64(0.07353154459753422), np.float64(0.22791878172588775)], n_bootstraps=10000, confidence_level=95)


    # Soil_nitrate

    Soil_nitrate_file_path = os.path.join(cur_path, 'Soil_nitrate', 'data.xlsx')
    cessation = pd.read_excel(Soil_nitrate_file_path, usecols='G,K').values[5:28, :]
    continous = pd.read_excel(Soil_nitrate_file_path, usecols='B,F').values[5:26, :]
    Soil_nitrate_k_interval, Soil_nitrate_rmse_interval, Soil_nitrate_R_squared_interval = bootstrap(cessation, continous, 'S', [-0.65, 0.2], n_bootstraps=10000, confidence_level=95)

    print(f'P_k_interval is {P_k_interval}')

    with open('bootstrap_runner_Result.txt', 'w') as file:
        file.write(f'BSA_low_k_interval is {BSA_low_k_interval} \n')
        file.write(f'BSA_low_rmse_interval is {BSA_low_rmse_interval} \n')
        file.write(f'BSA_intermediate_k_interval is {BSA_intermediate_k_interval} \n')
        file.write(f'BSA_intermediate_rmse_interval is {BSA_intermediate_rmse_interval} \n')
        file.write(f'BSA_high_k_interval is {BSA_high_k_interval} \n')
        file.write(f'BSA_high_rmse_interval is {BSA_high_rmse_interval} \n')
        file.write(f'P_k_interval is {P_k_interval} \n')
        file.write(f'P_loss_interval is {P_rmse_interval} \n')
        file.write(f'Soil_nitrate_k_interval is {Soil_nitrate_k_interval} \n')
        file.write(f'Soil_nitrate_rmse_interval is {Soil_nitrate_rmse_interval}')


import pandas as pd
import os
from data_fit.bootstrap import *

if __name__ == '__main__':
    cur_path = os.getcwd()


    # BSA

    BSA_file_path = os.path.join(cur_path, 'BSA', 'BSA_data.xlsx')
    data_BSA = pd.read_excel(BSA_file_path, header=1)

    data_BSA_low_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_low_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'low'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_intermediate_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'intermediate'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_intermediate_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'intermediate'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    data_BSA_high_recovery = data_BSA.loc[(data_BSA['state'] == 'recovery') & (data_BSA['enrichment treatment'] == 'high'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values
    data_BSA_high_enrichment = data_BSA.loc[(data_BSA['state'] == 'enrichment') & (data_BSA['enrichment treatment'] == 'high'), ['BSA(mg/ml)', 'dissolved oxygen(%)']].values

    BSA_low_k_interval, BSA_low_rmse_interval, BSA_low_R_squared_interval = bootstrap(data_BSA_low_recovery, data_BSA_low_enrichment, 'Z', [0.70, 1.1], n_bootstraps=10000, confidence_level=95)

    BSA_intermediate_k_interval, BSA_intermediate_rmse_interval, BSA_intermediate_R_squared_interval = bootstrap(data_BSA_intermediate_enrichment, data_BSA_intermediate_recovery, 'Z', [2.5, 5], n_bootstraps=10000, confidence_level=95)

    BSA_high_k_interval, BSA_high_rmse_interval, BSA_high_R_squared_interval = bootstrap(data_BSA_high_enrichment, data_BSA_high_recovery, 'Z', [5, 10], n_bootstraps=10000, confidence_level=95)


    # P

    P_file_path = os.path.join(cur_path, 'P', 'Pdata.xlsx')
    data1_0 = pd.read_excel(P_file_path, usecols='A:B', nrows=5).values
    data0_1 = pd.read_excel(P_file_path, usecols='D:E').values
    P_k_interval, P_rmse_interval, P_R_squared_interval = bootstrap(data0_1, data1_0, 'Z', [0.06, 0.15], n_bootstraps=10000, confidence_level=95)


    # Soil_nitrate

    Soil_nitrate_file_path = os.path.join(cur_path, 'Soil_nitrate', 'data.xlsx')
    cessation = pd.read_excel(Soil_nitrate_file_path, usecols='G,K').values[5:28, :]
    continous = pd.read_excel(Soil_nitrate_file_path, usecols='B,F').values[5:26, :]
    Soil_nitrate_k_interval, Soil_nitrate_rmse_interval, Soil_nitrate_R_squared_interval = bootstrap(cessation, continous, 'S', [-0.55, 0.1], n_bootstraps=10000, confidence_level=95)


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


import numpy as np
from S import *
from tqdm import trange


def bootstrap(data_0_1, data_1_0, shape, data_TPP_x, n_bootstraps=10000, confidence_level=95):
    n_samples_0_1 = len(data_0_1)
    n_samples_1_0 = len(data_1_0)
    bootstrap_k = []
    bootstrap_rmse = []
    bootstrap_R_squared = []

    for _ in trange(n_bootstraps):
        indices_0_1 = np.random.choice(n_samples_0_1, n_samples_0_1, replace=True)
        indices_1_0 = np.random.choice(n_samples_1_0, n_samples_1_0, replace=True)
        resampled_0_1 = data_0_1[indices_0_1]
        resampled_1_0 = data_1_0[indices_1_0]
        try:
            rmse, R_squared, k, x, y, pre_x_0_1, pre_x_1_0, attraction, att_derivative = fit_data(resampled_0_1, resampled_1_0, shape=shape, data_TPP_x=data_TPP_x)
            bootstrap_k.append(k)
            bootstrap_rmse.append(rmse)
            bootstrap_R_squared.append(R_squared)
        except RuntimeError:
            pass

    bootstrap_k = np.array(bootstrap_k)
    bootstrap_rmse = np.array(bootstrap_rmse)

    alpha = 100 - confidence_level
    lower_percentile = alpha / 2
    upper_percentile = 100 - alpha / 2

    k_ci_lower = np.percentile(bootstrap_k, lower_percentile)
    k_ci_upper = np.percentile(bootstrap_k, upper_percentile)

    rmse_ci_lower = np.percentile(bootstrap_rmse, lower_percentile)
    rmse_ci_upper = np.percentile(bootstrap_rmse, upper_percentile)

    R_squared_ci_lower = np.percentile(bootstrap_R_squared, lower_percentile)
    R_squared_ci_upper = np.percentile(bootstrap_R_squared, upper_percentile)

    return [k_ci_lower, k_ci_upper], [rmse_ci_lower, rmse_ci_upper], [R_squared_ci_lower, R_squared_ci_upper]






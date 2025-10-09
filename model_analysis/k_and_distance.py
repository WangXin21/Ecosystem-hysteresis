import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from S import *

cur_path = os.getcwd()
output_path = os.path.join(cur_path, 'Output')
k_list = np.linspace(6.33, 15, 1000)
fig, ax = plt.subplots()
lis = []
for k in k_list:
    model = ModelS(k=k)
    model.init()
    x, y = model.s()
    TP_1, TP_2 = model.find_TPP()
    x_1, x_2 = model.f(TP_1), model.f(TP_2)
    lis.append(abs(x_1-x_2))
    pre_x = liner_trans(x, 1, 2)
ax.plot(k_list, lis)
plt.legend()
plt.show()

k_and_distance = pd.DataFrame([k_list, lis], index=['k', 'distance between 2 TPPs']).T
k_and_distance.to_csv(os.path.join(output_path, 'k_and_distance.csv'))



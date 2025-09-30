import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from S import ModelS
import matplotlib as mpl


if __name__ == '__main__':
    s = ModelS()
    s.init()
    x, y = s.s()
    final = s.final_function(y, '0_1')
    data = [x, y, final, y]
    index = ['blue_x', 'blue_y', 'yellow_x', 'yellow_y']
    pd.DataFrame(data, index=index).T.to_csv('trans_curve.csv', index=False)
    pass

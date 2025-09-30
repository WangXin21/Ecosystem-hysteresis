import os

from S import *
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class ModelToData:
    def __init__(self,
                 data,
                 state,
                 shape,
                 k,
                 data_TPP_x,
                 weighted=False,
                 independent_variable=0,
                 dependent_variable=1,
                 c_0=1,
                 c_1=np.exp(1)):
        self.data_TPP_x = data_TPP_x
        data_y = data[:, dependent_variable]
        data_x = data[:, independent_variable]
        if shape == 'S':
            model = ModelS(k=k, c_0=c_0, c_1=c_1)
            self.model_x, self.model_y = model.s()
        else:
            model = ModelZ(k=k, c_0=c_0, c_1=c_1)
            self.model_x, self.model_y = model.z()
        model.init()

        TP_1, TP_2 = model.find_TPP()
        if state == '0_1':
            TP = copy.deepcopy(TP_2)
        else:
            TP = copy.deepcopy(TP_1)
        self.x_TP = model.f(TP)
        self.max_data_y, self.min_data_y = np.max(data_y), np.min(data_y)

        self.max_data_x = np.max(data_x)
        self.min_data_x = np.min(data_x)
        assert data_TPP_x < self.max_data_x, 'Data TP points are set too large'
        assert data_TPP_x > self.min_data_x, 'Data TP points are set too low'
        if weighted:
            count = 0
            for i in data_x:
                if i >= data_TPP_x:
                    count += 1
            self.w = count / len(data_x)
        else:
            self.w = 0.5

    def transform(self, m_point):
        x = m_point[0]
        y = m_point[1]
        data_y = line_map(y, [self.min_data_y, self.max_data_y], [np.min(self.model_y), np.max(self.model_y)])
        data_x = self.w * line_map(x, [self.max_data_x, self.data_TPP_x], [np.max(self.model_x), self.x_TP]) + (1 - self.w) * line_map(x, [self.min_data_x, self.data_TPP_x], [np.min(self.model_x), self.x_TP])
        return [data_x, data_y]

def prediction(data,
               state,
               pre_data,
               k,
               shape,
               data_TPP_x,
               weighted = False,
               independent_variable=0,
               dependent_variable=1,
               c_0=1,
               c_1=np.exp(1)
               ):
    if shape == 'S':
        model = ModelS(k=k, c_0=c_0, c_1=c_1)
        x, y = model.s()
    else:
        model = ModelZ(k=k, c_0=c_0, c_1=c_1)
        x, y = model.z()
    model.init()
    if state == '0_1':
        s = '1_0'
    else:
        s = '0_1'
    TP_1, TP_2 = model.find_TPP()
    if state == '0_1':
        TP = copy.deepcopy(TP_2)
    else:
        TP = copy.deepcopy(TP_1)
    x_TP = model.f(TP)
    MtD = ModelToData(data, state, shape, k, data_TPP_x, weighted)
    model_points = np.vstack([x, y]).T
    res = []
    for i in model_points:
        res.append(MtD.transform(i))
    pre_x, pre_y = np.array(res)[:, 0], np.array(res)[:, 1]
    if pre_data is not None:
        pre_data_x = pre_data[:, independent_variable]
        pre_data_y = pre_data[:, dependent_variable]
        n = len(pre_data_x)
        trans_pre_data_y = liner_trans(pre_data_y, 1 + 0.1, np.exp(1) - 0.1)
        trans_pre_x = np.array(model.final_function(trans_pre_data_y, state=s))
        loss_x = []
        for i in trans_pre_x:
            result = (line_map(i, [MtD.max_data_x, data_TPP_x], [np.max(x), x_TP]) + line_map(i, [MtD.min_data_x, data_TPP_x], [np.min(x), x_TP])) / 2
            loss_x.append(result)
        loss = np.linalg.norm(np.array(loss_x) - np.array(pre_data_x)) / np.sqrt(n)
    else:
        loss = 'Can not compute !'
    return loss, pre_x, pre_y


def get_prediction_path():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(cur_path, '..'))
    return os.path.join(root_path, 'Predict', 'Prediction')


import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt


def dichotomy(fun, interval, interation=50000, epsilon=0.01):
    """
    二分法求函数零点
    :param fun: 函数
    :param interval: 区间
    :param interation: 迭代上限
    :param epsilon: 精度
    :return: 返回零点
    """
    m = np.min(interval)
    M = np.max(interval)
    m_value = fun(m)
    if m_value == 0:
        return m
    M_value = fun(M)
    if M_value == 0:
        return M
    assert m_value * M_value < 0, 'Adjusting the range of intervals'

    mid = (m + M) / 2
    mid_value = fun(mid)
    if mid_value == 0:
        return mid
    i = 0
    while np.abs(mid_value) > epsilon:
        i += 1
        if i >= interation:
            return mid_value
        if mid_value * M_value < 0:
            m = mid
        else:
            M = mid
        mid = (m + M) / 2
        mid_value = fun(mid)
        if mid_value == 0:
            return mid
    return mid


class ModelS:
    """
    S型曲线
    """
    def __init__(self, c_0=1, c_1=np.exp(1), k=8):
        """
        :param c_0:
        :param c_1:
        :param k:
        """
        self.c_0 = c_0
        self.lower_bound = c_0 + 0.001
        self.upper_bound = c_1 - 0.1
        self.c_1 = c_1
        self.k = k
        self.k_star = 4 * self.c_1 * self.c_0 / (self.c_1 - self.c_0)
        # 求得tp点的y值
        a = self.c_1 - self.c_0 + self.k
        b = -self.k * (self.c_1 + self.c_0)
        c = self.k * self.c_0 * self.c_1
        delta = b ** 2 - 4 * a * c
        self.TP_1 = (-b + np.sqrt(delta)) / (2 * a)
        self.TP_2 = (-b - np.sqrt(delta)) / (2 * a)

        self.y_TP_1 = None
        self.y_TP_2 = None

    def init(self):
        # 利用二分法求得与tp点相同x坐标的点的y值
        self.y_TP_1 = dichotomy(self.fun_x_TP_1, [self.TP_2, self.c_0 + 0.001])
        self.y_TP_2 = dichotomy(self.fun_x_TP_2, [self.TP_1, self.c_1 - 0.001])

    def f(self, y):
        """
        模型函数
        :param y:
        :return:
        """
        return (y - self.c_0) / (self.c_1 - y) * np.exp(self.k / y)

    def fun_x_TP_2(self, y):
        # 二分法的函数
        TP_1, TP_2 = self.find_TPP()
        x = self.f(TP_2)
        return self.f(y) - x

    def fun_x_TP_1(self, y):
        # 二分法的函数
        TP_1, TP_2 = self.find_TPP()
        x = self.f(TP_1)
        return self.f(y) - x

    def get_y_TP_1(self):
        return self.y_TP_1

    def get_y_TP_2(self):
        return self.y_TP_2

    def s(self):
        # 返回模型的x，y值
        y = np.linspace(self.c_0+0.001, self.c_1-0.001, 1000)
        return self.f(y), y

    def get_att(self, y):
        # 取对数作为吸引力
        return [self.k / y, np.log((y - self.c_0) / (self.c_1 - y))]

    def get_att_derivative(self, y):
        # 吸引力导数
        return [-self.k / (y ** 2), 1 / (y - self.c_0) + 1 / (self.c_1 - y)]

    def find_TPP(self):
        return self.TP_1, self.TP_2

    def final_function(self, y, state):
        # 带有迟滞效应的函数
        TP_1, TP_2 = self.find_TPP()
        y_TP_2 = self.y_TP_2
        y_TP_1 = self.y_TP_1
        result = []
        if state == '0_1':
            for i in y:
                if i < TP_2:
                    result.append(self.f(i))
                elif i > y_TP_2:
                    result.append(self.f(i))
                else:
                    result.append(self.f(TP_2))
        else:
            for i in y:
                if i > TP_1:
                    result.append(self.f(i))
                elif i < y_TP_1:
                    result.append(self.f(i))
                else:
                    result.append(self.f(TP_1))
        return result


class ModelZ:
    def __init__(self, c_0=1, c_1=np.exp(1), k=8):
        self.c_0 = c_0
        self.c_1 = c_1
        self.lower_bound = c_0 + 0.1
        self.upper_bound = c_1 - 0.001
        self.k = k
        self.k_star = 4 * self.c_1 * self.c_0 / (self.c_1 - self.c_0)
        s = ModelS(self.c_0, self.c_1, self.k)
        s_TP_1, s_TP_2 = s.find_TPP()
        self.TP_1 = self.c_1 + self.c_0 - s_TP_2
        self.TP_2 = self.c_1 + self.c_0 - s_TP_1
        self.y_TP_1 = None
        self.y_TP_2 = None

    def f(self, y):
        return (self.c_1 - y)/(y - self.c_0) * np.exp(self.k / (self.c_1 + self.c_0 - y))

    def z(self):
        s = ModelS(self.c_0, self.c_1, self.k)
        x, y = s.s()
        return x, self.c_1 + self.c_0 - y

    def get_att(self, y):
        return [np.log((self.c_1 - y)/(y - self.c_0)), self.k / (self.c_1 + self.c_0 - y)]

    def get_att_derivative(self, y):
        return [-1 / (y - self.c_0) - 1 / (self.c_1 - y), (self.k / (self.c_1 + self.c_0 - y) ** 2)]

    def find_TPP(self):
        return self.TP_1, self.TP_2

    def init(self):
        s = ModelS(self.c_0, self.c_1, self.k)
        s.init()
        y_TP_2, y_TP_1 = s.get_y_TP_2(), s.get_y_TP_1()
        self.y_TP_1 = self.c_1 + self.c_0 - y_TP_2
        self.y_TP_2 = self.c_1 + self.c_0 - y_TP_1

    def get_y_TP_1(self):
        return self.y_TP_1

    def get_y_TP_2(self):
        return self.y_TP_2

    def final_function(self, y, state):
        TP_1, TP_2 = self.find_TPP()
        y_TP_2, y_TP_1 = self.get_y_TP_2(), self.get_y_TP_1()
        result = []
        if state == '0_1':
            for i in y:
                if i < TP_2:
                    result.append(self.f(i))
                elif i > y_TP_2:
                    result.append(self.f(i))
                else:
                    result.append(self.f(TP_2))
        else:
            for i in y:
                if i > TP_1:
                    result.append(self.f(i))
                elif i < y_TP_1:
                    result.append(self.f(i))
                else:
                    result.append(self.f(TP_1))
        return result


def liner_trans(data, min, max):
    """
    将数据线性变换到指定范围
    :param data:
    :param min:
    :param max:
    :return:
    """
    min_data = np.min(data)
    max_data = np.max(data)
    k = (max - min)/(max_data - min_data)
    y = k * (data - min_data) + min
    return y


def line_map(x, target, Preimage):
    # 将数据按某一线性映射做变换
    k = (np.max(target) - np.min(target)) / (np.max(Preimage) - np.min(Preimage))
    y = k * (x - np.min(Preimage)) + np.min(target)
    return y


def trans(data_state0_1,
          data_state1_0,
          k,
          shape,
          data_TPP_x = None,
          independent_variable=0,
          dependent_variable=1,
          c_0=1,
          c_1=np.exp(1)
          ):
    """
    计算当前k值下损失的函数
    :param data_state0_1: 状态0-1的数据
    :param data_state1_0: 状态1-0的数据
    :param k:
    :param shape: ‘S’或‘Z’
    :param method: 线性映射或者标准化后再指数运算
    :param data_TPP_x: 数据的tp点坐标
    :param independent_variable: 自变量列数
    :param dependent_variable:
    :param c_0:
    :param c_1:
    :return: 当前k值下损失
    """
    if shape == 'S':
        model = ModelS(k=k)

    else:
        model = ModelZ(k=k)
    model.init()
    data_x_0_1 = data_state0_1[:, independent_variable]
    data_x_1_0 = data_state1_0[:, independent_variable]
    data_y_0_1 = data_state0_1[:, dependent_variable]
    data_y_1_0 = data_state1_0[:, dependent_variable]
    data_x = np.append(data_x_0_1, data_x_1_0)
    n = len(data_x)
    data_x_min, data_x_max = np.min(data_x), np.max(data_x)
    len_state0_1 = len(data_x_0_1)
    len_state1_0 = len(data_x_1_0)
    data_y = np.append(data_y_0_1, data_y_1_0)
    trans_data_y = liner_trans(data_y, model.lower_bound, model.upper_bound)
    trans_data_y_0_1 = trans_data_y[:len_state0_1]
    trans_data_y_1_0 = trans_data_y[-len_state1_0:]
    x_0_1 = np.array(model.final_function(trans_data_y_0_1, state='0_1'))
    x_1_0 = np.array(model.final_function(trans_data_y_1_0, state='1_0'))
    TP_1, TP_2 = model.find_TPP()
    TP_1_x, TP_2_x = model.f(TP_1), model.f(TP_2)
    if data_TPP_x is None:

        condition_0_1 = (trans_data_y_0_1 >= TP_2) & (trans_data_y_0_1<= model.y_TP_2)
        tipping_data_x_0_1 = data_x_0_1[condition_0_1]
        if len(tipping_data_x_0_1) > 0:
            data_TPP_x_0_1 = np.mean(tipping_data_x_0_1)
        else:
            data_TPP_x_0_1 = np.mean(data_x_0_1)

        condition_1_0 = (trans_data_y_1_0 <= TP_1) & (trans_data_y_1_0 >= model.y_TP_1)
        tipping_data_x_1_0 = data_x_1_0[condition_1_0]
        if len(tipping_data_x_1_0) > 0:
            data_TPP_x_1_0 = np.mean(tipping_data_x_1_0)
        else:
            data_TPP_x_1_0 = np.mean(data_x_1_0)

        data_TPP_x = [data_TPP_x_0_1, data_TPP_x_1_0]
    target = data_TPP_x
    Preimage = [TP_1_x, TP_2_x]
    pre_x_0_1 = line_map(x_0_1, target, Preimage)
    pre_x_1_0 = line_map(x_1_0, target, Preimage)

    # else:
    #     target = data_TPP_x
    #     Preimage = [TP_1_x, TP_2_x]
    #     pre_x_0_1 = line_map(x_0_1, target, Preimage)
    #     pre_x_1_0 = line_map(x_1_0, target, Preimage)
    loss_1 = np.linalg.norm(pre_x_0_1-data_x_0_1) ** 2
    loss_2 = np.linalg.norm(pre_x_1_0-data_x_1_0) ** 2
    loss = loss_1 + loss_2
    R_squared = 1 - loss / (n * np.var(data_x))
    return np.sqrt(loss / n), R_squared


def fit_data(data_state_0_1,
             data_state_1_0,
             shape,
             k=None,
             data_TPP_x = None,
             c_0=1,
             c_1=np.exp(1),
             independent_variable=0,
             dependent_variable=1
             ):
    """
    寻找最优k值，参数同上
    :param data_state_0_1:
    :param data_state_1_0:
    :param shape:
    :param k:
    :param method:
    :param data_TPP_x:
    :param c_0:
    :param c_1:
    :param independent_variable:
    :param dependent_variable:
    :return:
    """
    if k is None:
        k_star = 4 * c_1 * c_0 / (c_1 - c_0)
        k_list = np.linspace(k_star+0.01, 15, 2000)
        # pbar = tqdm(k_list)
        # pbar.set_description('Processing:')
        rmse_list = []
        R_squared_list = []
        for k_ in k_list:
            loss, R_squared = trans(data_state_0_1, data_state_1_0, k_, shape, data_TPP_x)
            rmse_list.append(loss)
            R_squared_list.append(R_squared)
        best_index = np.argmin(rmse_list)
        best_k = k_list[best_index]
        best_R_squared = R_squared_list[best_index]
        best_rmse = rmse_list[best_index]
    else:
        best_k = k
        best_rmse, best_R_squared = trans(data_state_0_1, data_state_1_0, best_k, shape, data_TPP_x)
    if shape == 'S':
        model = ModelS(k=best_k)

        x, y = model.s()
    else:
        model = ModelZ(k=best_k)
        x, y = model.z()
    model.init()
    data_x_0_1 = data_state_0_1[:, independent_variable]
    data_x_1_0 = data_state_1_0[:, independent_variable]
    data_y_0_1 = data_state_0_1[:, dependent_variable]
    data_y_1_0 = data_state_1_0[:, dependent_variable]
    data_y = np.append(data_y_0_1, data_y_1_0)
    x_0_1 = np.array(model.final_function(y, state='0_1'))
    x_1_0 = np.array(model.final_function(y, state='1_0'))
    attraction = model.get_att(y)
    attraction_derivative = model.get_att_derivative(y)
    TP_1, TP_2 = model.find_TPP()
    TP_1_x, TP_2_x = model.f(TP_1), model.f(TP_2)
    len_state0_1 = len(data_x_0_1)
    len_state1_0 = len(data_x_1_0)
    trans_data_y = liner_trans(data_y, model.lower_bound, model.upper_bound)
    trans_data_y_0_1 = trans_data_y[:len_state0_1]
    trans_data_y_1_0 = trans_data_y[-len_state1_0:]
    if data_TPP_x is None:

        condition_0_1 = (trans_data_y_0_1 >= TP_2) & (trans_data_y_0_1 <= model.y_TP_2)
        tipping_data_x_0_1 = data_x_0_1[condition_0_1]
        if len(tipping_data_x_0_1) > 0:
            data_TPP_x_0_1 = np.mean(tipping_data_x_0_1)
        else:
            data_TPP_x_0_1 = np.mean(data_x_0_1)

        condition_1_0 = (trans_data_y_1_0 <= TP_1) & (trans_data_y_1_0 >= model.y_TP_1)
        tipping_data_x_1_0 = data_x_1_0[condition_1_0]
        if len(tipping_data_x_1_0) > 0:
            data_TPP_x_1_0 = np.mean(tipping_data_x_1_0)
        else:
            data_TPP_x_1_0 = np.mean(data_x_1_0)

        data_TPP_x = [data_TPP_x_0_1, data_TPP_x_1_0]
    target = data_TPP_x
    Preimage = [TP_1_x, TP_2_x]
    pre_x_0_1 = line_map(x_0_1, target, Preimage)
    pre_x_1_0 = line_map(x_1_0, target, Preimage)
    pre_x = line_map(x, target, Preimage)
    # if data_TPP_x is None:
    #     pre_x = liner_trans(x, data_x_min, data_x_max)
    #     pre_x_0_1 = liner_trans(x_0_1, data_x_min, data_x_max)
    #     pre_x_1_0 = liner_trans(x_1_0, data_x_min, data_x_max)
    # else:
    #     TP_1, TP_2 = model.find_TPP()
    #     TP_1_x, TP_2_x = model.f(TP_1), model.f(TP_2)
    #     target = data_TPP_x
    #     Preimage = [TP_1_x, TP_2_x]
    #     pre_x = line_map(x, target, Preimage)
    #     pre_x_0_1 = line_map(x_0_1, target, Preimage)
    #     pre_x_1_0 = line_map(x_1_0, target, Preimage)
    target_y = data_y
    Preimage_y = [model.upper_bound, model.lower_bound]
    pre_y = line_map(y, target_y, Preimage_y)
    return best_rmse, best_R_squared, best_k, data_TPP_x, pre_x, pre_y, pre_x_0_1, pre_x_1_0, attraction, attraction_derivative

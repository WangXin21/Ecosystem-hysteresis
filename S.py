import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt


def dichotomy(fun, interval, interation=50000, epsilon=0.001):
    """
    Find a root of a function within a given interval using the dichotomy method.
    The dichotomy method is an iterative root-finding algorithm that repeatedly bisects an interval and selects a
    subinterval in which a root must lie for further processing.
     Parameters
    ----------
    fun : callable
        A function for which to find the root. Must accept a single float argument
        and return a float value.
    interval : list or tuple
        A sequence of two numbers representing the initial interval [a, b] to search
        for a root. The function must have opposite signs at these endpoints.
    interation : int, optional
        Maximum number of iterations to perform (default is 50000). Prevents
        infinite loops in case of slow convergence.
    epsilon : float, optional
        Desired accuracy tolerance. The algorithm stops when |f(mid)| < epsilon
        (default is 0.01).

    Returns
    -------
    float
        An approximation of the root of the function within the given interval.
        If maximum iterations are reached, returns the current function value at
        the midpoint instead.
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
    S Model
    """
    def __init__(self, c_0=1, c_1=np.exp(1), k=9):
        self.c_0 = c_0
        self.c_1 = c_1
        self.k = k
        self.k_star = 4 * self.c_1 * self.c_0 / (self.c_1 - self.c_0)
        # SSet boundaries for the simulation while avoiding numerical explosion
        self.lower_bound = c_0 + 0.001
        self.upper_bound = c_1 - 0.1
        self.TP_1 = None
        self.TP_2 = None
        self.y_TP_1 = None
        self.y_TP_2 = None


    def init(self):
        # the landing point after regime shift is obtained by dichotomy
        if self.k > self.k_star:
            # Obtain the y value of the tipping-points, self.TP_1 and self.TP_2, and self.TP_1 > self.TP_2
            a = self.c_1 - self.c_0 + self.k
            b = -self.k * (self.c_1 + self.c_0)
            c = self.k * self.c_0 * self.c_1
            delta = b ** 2 - 4 * a * c
            self.TP_1 = (-b + np.sqrt(delta)) / (2 * a)
            self.TP_2 = (-b - np.sqrt(delta)) / (2 * a)
            self.y_TP_1 = dichotomy(self.fun_x_TP_1, [self.TP_2, self.c_0 + 0.001])
            self.y_TP_2 = dichotomy(self.fun_x_TP_2, [self.TP_1, self.c_1 - 0.001])

    def f(self, y):
        return (y - self.c_0) / (self.c_1 - y) * np.exp(self.k / y)

    def fun_x_TP_2(self, y):
        TP_1, TP_2 = self.find_TPP()
        x = self.f(TP_2)
        return self.f(y) - x

    def fun_x_TP_1(self, y):
        TP_1, TP_2 = self.find_TPP()
        x = self.f(TP_1)
        return self.f(y) - x

    def get_y_TP_1(self):
        return self.y_TP_1

    def get_y_TP_2(self):
        return self.y_TP_2

    def s(self):
        # Return the numerical simulation results
        y = np.linspace(self.c_0+0.001, self.c_1-0.001, 1000)
        return self.f(y), y

    def get_att(self, y):
        # Take the logarithm as the attractive force
        return [self.k / y, np.log((y - self.c_0) / (self.c_1 - y))]

    def get_att_derivative(self, y):
        # The derivative of attractive force
        return [-self.k / (y ** 2), 1 / (y - self.c_0) + 1 / (self.c_1 - y)]

    def find_TPP(self):
        return self.TP_1, self.TP_2

    def final_function(self, y, state):
        # The function of regime shift
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
    def __init__(self, c_0=1, c_1=np.exp(1), k=9):
        self.c_0 = c_0
        self.c_1 = c_1
        self.lower_bound = c_0 + 0.1
        self.upper_bound = c_1 - 0.001
        self.k = k
        self.k_star = 4 * self.c_1 * self.c_0 / (self.c_1 - self.c_0)
        self.TP_1 = None
        self.TP_2 = None
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
        s_TP_1, s_TP_2 = s.find_TPP()
        if s_TP_1 is not None:
            self.TP_1 = self.c_1 + self.c_0 - s_TP_2
            self.TP_2 = self.c_1 + self.c_0 - s_TP_1
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
    # Linearly transform the data to the specified range
    min_data = np.min(data)
    max_data = np.max(data)
    k = (max - min)/(max_data - min_data)
    y = k * (data - min_data) + min
    return y


def line_map(x, target, Preimage):
    # Transform the data according to a certain linear mapping
    k = (np.max(target) - np.min(target)) / (np.max(Preimage) - np.min(Preimage))
    y = k * (x - np.min(Preimage)) + np.min(target)
    return y


def loss_compute(data_state0_1,
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
    Calculate the functions of RMSE and R-squared under a certain k
    :param data_state0_1: Data transferred from state C_0 to state C_1
    :param data_state1_0: Data transferred from state C_1 to state C_2
    :param k:
    :param shape: ‘S’或‘Z’
    :param data_TPP_x: The X-coordinate of the tp point of the data can also be left None.
                       If left None, the algorithm will calculate it automatically
    :param independent_variable: In which column is the driving factor of the data located
    :param dependent_variable: In which column is the state of the data located
    """
    if shape == 'S':
        model = ModelS(k=k)

    else:
        model = ModelZ(k=k)
    model.init()
    data_x_0_1 = data_state0_1[:, independent_variable].astype(float)
    data_x_1_0 = data_state1_0[:, independent_variable].astype(float)
    data_y_0_1 = data_state0_1[:, dependent_variable].astype(float)
    data_y_1_0 = data_state1_0[:, dependent_variable].astype(float)
    data_x = np.append(data_x_0_1, data_x_1_0)
    n = len(data_x)
    data_x_min, data_x_max = np.min(data_x), np.max(data_x)
    len_state0_1 = len(data_x_0_1)
    len_state1_0 = len(data_x_1_0)
    data_y = np.append(data_y_0_1, data_y_1_0)
    trans_data_y = liner_trans(data_y, model.lower_bound, model.upper_bound)
    trans_data_y_0_1 = trans_data_y[:len_state0_1]
    trans_data_y_1_0 = trans_data_y[-len_state1_0:]
    if model.TP_1 is not None:
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
    else:
        x_0_1 = model.f(trans_data_y_0_1)
        x_1_0 = model.f(trans_data_y_1_0)
        target = [np.min(data_x), np.max(data_x)]
        Preimage = [model.f(model.lower_bound), model.f(model.upper_bound)]
        pre_x_0_1 = line_map(x_0_1, target, Preimage)
        pre_x_1_0 = line_map(x_1_0, target, Preimage)
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
    Find the optimal k value with the same parameters as above
    """
    if k is None:
        k_star = 4 * c_1 * c_0 / (c_1 - c_0)
        k_list = np.linspace(0, 15, 15000)
        rmse_list = []
        R_squared_list = []
        for k_ in k_list:
            loss, R_squared = loss_compute(data_state_0_1, data_state_1_0, k_, shape, data_TPP_x)
            rmse_list.append(loss)
            R_squared_list.append(R_squared)
        best_index = np.argmin(rmse_list)
        best_k = k_list[best_index]
        best_R_squared = R_squared_list[best_index]
        best_rmse = rmse_list[best_index]
    else:
        best_k = k
        best_rmse, best_R_squared = loss_compute(data_state_0_1, data_state_1_0, best_k, shape, data_TPP_x)
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
    data_x = np.append(data_x_0_1, data_x_1_0)
    data_y = np.append(data_y_0_1, data_y_1_0)
    if model.TP_1 is not None:
        x_0_1 = np.array(model.final_function(y, state='0_1'))
        x_1_0 = np.array(model.final_function(y, state='1_0'))
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
    else:
        target = [np.min(data_x), np.max(data_x)]
        Preimage = [model.f(model.lower_bound), model.f(model.upper_bound)]
        pre_x = line_map(x, target, Preimage)
        pre_x_0_1 = line_map(x, target, Preimage)
        pre_x_1_0 = line_map(x, target, Preimage)
    attraction = model.get_att(y)
    attraction_derivative = model.get_att_derivative(y)
    target_y = data_y
    Preimage_y = [model.upper_bound, model.lower_bound]
    pre_y = line_map(y, target_y, Preimage_y)
    return best_rmse, best_R_squared, best_k, data_TPP_x, pre_x, pre_y, pre_x_0_1, pre_x_1_0, attraction, attraction_derivative

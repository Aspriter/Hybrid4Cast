import pdb
import warnings
import keras.backend as K
from inspect import signature
import numpy as np


def print_section_msg(msg):
    print('---------- {0} ---------- '.format(msg))


def parse_int(int_str):
    try:
        return int(int_str)
    except ValueError:
        return None


def value_sampler(x, method='median'):
    if method == 'mean':
        return np.mean(x)
    elif method == 'upmed':
        return np.percentile(x, q=60)
    elif method == 'interq':
        return np.mean(np.percentile(x, q=[25, 75]))
    else:
        return np.median(x)


def array_division(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError('the array division has to have the same length!')
    if len(arr1) == 0:
        return arr1

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = [0 if np.abs(v2) < 1e-8 else v1/v2 for v1, v2 in zip(arr1, arr2)]
        except Warning as e:
            pdb.set_trace()

    return result


def sample_concatenate(samples, axis=0):
    if len(samples) == 0:
        return None
    else:
        return np.concatenate(samples, axis=axis)


def smape_log(y_true, y_pred):
    y_pred = K.exp(y_pred) - 1
    y_true = K.exp(y_true) - 1
    smape = 2 * K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true) + 1e-16)
    smape = K.mean(smape, 1)
    smape = K.mean(smape, 1)
    smape = K.mean(smape) * 100
    return smape


def get_smape_error(pred, real):
    eps = 1e-16  # Need to make sure that denominator is not zero
    norm = 0.5 * (np.abs(pred) + np.abs(real)) + eps
    return np.round(np.mean(np.abs(pred - real) / norm) * 100.0, 2)


def show_value_quantiles(values, perc=None):
    np.set_printoptions(suppress=True)
    if perc is None:
        perc = list(range(0, 80, 10))
        perc.append(100)

    valid_idx = np.where(np.isfinite(values))[0]
    if len(valid_idx) != len(values):
        print('there is {0} NaN values'.format(len(values)-len(valid_idx)))

    print(np.around(np.nanpercentile(values, q=perc), decimals=2))


def show_nn_sample_value_quantiles(x_train=None, y_train=None, x_val=None, y_val=None, percentiles=[0, 5, 25, 50, 75, 95, 100]):
    matrix_names = ['train input x', 'train output y', 'validation input x', 'validation output y']
    data_matrix_list = (x_train, y_train, x_val, y_val)
    for i in range(len(data_matrix_list)):
        if data_matrix_list[i] is not None:
            print('value quantiles for {0}; shape {1}:'.format(matrix_names[i], data_matrix_list[i].shape))
            show_3d_matrix_value_quantile(data_matrix_list[i], percentiles)


def show_3d_matrix_value_quantile(matrix, percentiles=[0, 5, 25, 50, 75, 95, 100]):
    np.set_printoptions(suppress=True)
    for i in range(matrix.shape[2]):
        print("dimension {0}: {1}".format(i, np.around(np.percentile(matrix[:, :, i], q=percentiles), decimals=3)))


def do_para_call(func, *args, **kwargs):
    arg_dict = signature(func)
    para_list = list(arg_dict.parameters.keys())

    this_kwarg = kwargs
    if 'kwargs' not in para_list:
        this_kwarg = dict()
        for k in kwargs.keys():
            if k in para_list:
                this_kwarg[k] = kwargs[k]

    ba = arg_dict.bind(*args, **this_kwarg)
    ba.apply_defaults()
    return func(*ba.args, **ba.kwargs)

import pdb
import numpy as np


class Scaler:

    inter_quantile = [25, 75]
    outlier_range_multiplyer = 1.5

    def __init__(self, data_matrix, outlier_colidx):
        self.mean = None
        self.std = None

        self.compute_scaler(data_matrix, outlier_colidx)

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def compute_scaler(self, data_matrix, outlier_colidx):
        if outlier_colidx is None:
            self.mean = np.mean(data_matrix, axis=0)
            self.std = np.std(data_matrix, axis=0)
        else:
            data_matrix_mean = np.zeros(data_matrix.shape[1:])
            data_matrix_std = data_matrix_mean.copy()
            for i in range(data_matrix.shape[1]):
                for j in range(data_matrix.shape[2]):
                    if j in outlier_colidx:
                        valor = self.outlier_filter(data_matrix[:, i, j])
                    else:
                        valor = data_matrix[:, i, j]

                    if len(valor) == 0:
                        raise ValueError('there is 0 value left after removing outliers')

                    data_matrix_mean[i, j] = np.mean(valor)
                    data_matrix_std[i, j] = np.std(valor)

            self.mean = data_matrix_mean
            self.std = data_matrix_std

    @staticmethod
    def outlier_filter(values):
        inter_quantile = np.percentile(values, q=Scaler.inter_quantile)
        inter_quantile_range = inter_quantile[1] - inter_quantile[0]
        center = np.mean(inter_quantile)
        return values[np.where(abs(values - center) <= (0.5 + Scaler.outlier_range_multiplyer) * inter_quantile_range)]


class HybridModelNormalizer:

    def __init__(self, train_x, train_y, outlier_colidx_x=None, outlier_colidx_y=None, threshold=None, method='neutral'):
        if len(train_x.shape) != 3:
            raise Exception('The normalizer can only work for 3-dimension data matrix')

        self.outlier_colidx_x = outlier_colidx_x
        self.outlier_colidx_y = outlier_colidx_y
        self.threshold = threshold
        if method not in ['drop', 'trim', 'neutral']:
            raise Exception('unknown outlier handling method parameter!')
        self.method = method

        self.scaler_x = Scaler(train_x, outlier_colidx_x)
        self.scaler_y = Scaler(train_y, outlier_colidx_y)

    @staticmethod
    def rescale_one_matrix(data_matrix, scaler: Scaler):
        return data_matrix * scaler.get_std() + scaler.get_mean()

    @staticmethod
    def normalize_one_matrix(data_matrix, scaler: Scaler, outlier_colidx=None, threshold=None, method=None):

        # force set the sd whose value is 0 as 1, so that division wont get Inf
        data_matrix_std = scaler.get_std()
        zidx = np.argwhere(data_matrix_std == 0)
        if len(zidx) != 0:
            data_matrix_std[zidx[:, 0], zidx[:, 1]] = 1
        nor_data_matrix = (data_matrix - scaler.get_mean()) / data_matrix_std

        num_sample = data_matrix.shape[0]
        legit_sample_index = list(range(num_sample))
        outlier_replacement = threshold
        if method == 'neutral':
            outlier_replacement = 0
        if threshold is not None and outlier_colidx is not None:
            if method in ['trim', 'neutral']:
                for i in range(num_sample):
                    for j in outlier_colidx:
                        exvidx = np.where(np.abs(nor_data_matrix[i, :, j]) > threshold)[0]
                        nor_data_matrix[i, exvidx, j] = np.sign(nor_data_matrix[i, exvidx, j]) * outlier_replacement
            else:
                legit_sample_index = [len(np.where(np.abs(onesample[:, outlier_colidx]) > threshold)[0]) for onesample in nor_data_matrix]
                legit_sample_index = np.where(np.array(legit_sample_index) < 1)[0]

        return nor_data_matrix, legit_sample_index

    def normalize_xy_pair(self, xy_pair, outlier_colidx_x=None, outlier_colidx_y=None, threshold=None, method=None):
        if len(xy_pair) != 2 or len(xy_pair[0]) < 1:
            raise Exception('This X/Y pair format is wrong!')
        x_input, y_output = xy_pair

        outlier_colidx_x = self.outlier_colidx_x if outlier_colidx_x is None else outlier_colidx_x
        outlier_colidx_y = self.outlier_colidx_y if outlier_colidx_y is None else outlier_colidx_y
        threshold = self.threshold if threshold is None else threshold
        method = self.method if method is None else method

        x_normed, x_legit_index = self.normalize_one_matrix(x_input, self.scaler_x, outlier_colidx_x, threshold, method)
        y_normed, y_legit_index = self.normalize_one_matrix(y_output, self.scaler_y, outlier_colidx_y, threshold, method)

        # if it is drop mode, discard those samples with extreme values
        if threshold is not None and method == 'drop':
            valid_index = np.intersect1d(x_legit_index, y_legit_index)
            x_normed = x_normed[valid_index, :, :]
            y_normed = y_normed[valid_index, :, :]

        return x_normed, y_normed


import abc
from ..classicts.ets_forecaster import ETSForecaster
from .hybrid_samples import ETSNNSampleGenerator
from ..utils import show_nn_sample_value_quantiles


class HybridPipeline(object):

    sample_hislen_multiplier = 4

    def __init__(self, sample_horizon, sample_hislen=140, season_period=7, use_transform='log', damped_ets=True,
                 base_season_policy=None, long_season_threshold=None, long_season_policy=None):

        self.sample_horizon = sample_horizon
        self.sample_hislen = HybridPipeline.sample_hislen_multiplier * sample_horizon \
            if sample_hislen is None else sample_hislen

        self.use_transform = use_transform

        self.season_period = season_period
        self.long_season_threshold = long_season_threshold
        self.damped_ets = damped_ets

        self.base_season_policy = base_season_policy
        self.long_season_policy = long_season_policy

        self.normalizer = None
        self.cnn_model = None

    def set_hybrid_sample_history_length(self, sample_hislen):
        self.sample_hislen = sample_hislen

    @staticmethod
    @abc.abstractmethod
    def get_etsfit_statz(etsfit_list):
        pass

    def generate_etsfits(self, ts_matrix):
        num_target = ts_matrix.shape[1]

        target_etsfits = list()
        for i in range(num_target):
            train_data = ts_matrix[:, i]
            etsfit = ETSForecaster(
                train_data, self.sample_horizon, self.season_period,
                self.long_season_threshold, self.damped_ets, self.use_transform
            )
            target_etsfits.append(etsfit)

        hislen = ETSNNSampleGenerator.get_etsfit_pooled_length(target_etsfits)
        for i in range(num_target):
                etsfit = target_etsfits[i]
                etsfit.update_ts_length(hislen)
                etsfit.fit_grid_search()

        return target_etsfits

    @abc.abstractmethod
    def ets_train_model(self, input_data, **kwargs):
        pass

    @abc.abstractmethod
    def ets_make_forecast(self, etsfit_list, **kwargs):
        pass

    @abc.abstractmethod
    def generate_training_samples(self, etsfit_list, holdout_length=0, train_stride=None, num_val=1,
                                  num_train=None, train_random=True, **kwargs):
        pass

    @abc.abstractmethod
    def normalize_nn(self, x_train, y_train, x_val=None, y_val=None,
                     outlier_colidx_x=None, outlier_colidx_y=None, threshold=None, method='neutral'):
        pass

    @abc.abstractmethod
    def train_nn_model(self, x_train, y_train, x_val=None, y_val=None, num_filters=32, num_layers=2,
                       dropout_rate=(0.1, 0.3), loss='mae', learning_rate=1e-3, optimizer='adam',
                       batch_size=None, epochs=50, best_model_temp_filename=None, verbose=0, **kwargs):
        pass

    def hybrid_train_model(self, etsfit_list, holdout_length=0, train_stride=None, num_val=1, num_train=None, train_random=True,
                           outlier_colidx_x=None, outlier_colidx_y=None, threshold=5, method='neutral',
                           num_filters=32, num_layers=2, dropout_rate=(0.1, 0.3), loss='mae',
                           learning_rate=1e-3, optimizer='adam', batch_size=None, epochs=50, use_callback=False,
                           best_model_temp_filename=None, verbose_control=1, verbose_nn=0):
        if verbose_control:
            print('generating train/(val) samples')
        x_train_set, y_train_set, x_val_set, y_val_set = \
            self.generate_training_samples(etsfit_list, holdout_length, train_stride, num_val,
                                           num_train, train_random, verbose=verbose_control)

        if verbose_control:
            print('start normalizing samples')
            percentiles = [0, 5, 25, 50, 75, 95, 100]
            print('raw data value quantiles {0}:'.format(percentiles))
            show_nn_sample_value_quantiles(x_train_set, y_train_set, x_val_set, y_val_set, percentiles)

        x_train_set, y_train_set, x_val_set, y_val_set = \
            self.normalize_nn(x_train_set, y_train_set, x_val_set, y_val_set,
                              outlier_colidx_x, outlier_colidx_y, threshold, method)

        if verbose_control:
            print('nn normalized data value quantiles {0}:'.format(percentiles))
            show_nn_sample_value_quantiles(x_train_set, y_train_set, x_val_set, y_val_set, percentiles)

        # train model locally
        if verbose_control:
            print('training cnn model')
        self.train_nn_model(
            x_train_set, y_train_set, x_val_set, y_val_set,
            num_filters, num_layers, dropout_rate, loss,
            learning_rate, optimizer, batch_size, epochs, use_callback,
            best_model_temp_filename, verbose_nn
        )

    @abc.abstractmethod
    def get_forecast_input(self, etsfit_list, holdout_length=0):
        pass

    @abc.abstractmethod
    def get_actual_output(self, etsfit_list, holdout_length, **kwargs):
        pass

    @abc.abstractmethod
    def hybrid_make_forecast(self, x_forecast_input, y_ets_normalizer=None, y_value_transformer=None, **kwargs):
        pass

    @abc.abstractmethod
    def compute_smape(self, y_prediction, y_actual, verbose=1, **kwargs):
        pass

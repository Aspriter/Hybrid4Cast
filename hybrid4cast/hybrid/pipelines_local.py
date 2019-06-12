import pdb
import pickle
import warnings
import numpy as np
import pandas as pd
from numbers import Number
from tqdm import tqdm

from . import HybridPipeline

from ..cnn.hybrid_model_forecaster import HybridModelForecaster
from ..hybrid.hybrid_samples import ETSNNSampleGenerator
from ..hybrid.normalizer import HybridModelNormalizer
from ..utils import sample_concatenate, get_smape_error, show_value_quantiles


class HybridModelPipelineLocal(HybridPipeline):

    def __init__(self, sample_horizon=35, sample_hislen=140, season_period=7, use_transform='log', damped_ets=True,
                 base_season_policy=None, long_season_threshold=None, long_season_policy=None, use_ext_regr=False):

        self.hybrid_sample_generators = None

        super(HybridModelPipelineLocal, self).__init__(
            sample_horizon, sample_hislen, season_period, use_transform, damped_ets,
            base_season_policy, long_season_threshold, long_season_policy, use_ext_regr
        )

    @staticmethod
    def get_etsfit_statz(etsfit_list):
        progress_bar = tqdm(total=len(etsfit_list), desc='processing', unit=' cpg', leave=True, ascii=True, ncols=105)

        stat_table = list()
        for one_item in etsfit_list:
            progress_bar.update(1)

            etsfit_targets = one_item[1]
            one_row = [one_item[0]]
            for etsfit in etsfit_targets:
                values = np.expm1(etsfit.proc_ts.values)
                one_row = np.concatenate([one_row, [len(values), np.mean(values), np.std(values)]])
                one_row = np.concatenate([one_row, etsfit.best_config[:-1]])

            stat_table.append(one_row)

        progress_bar.close()
        return pd.DataFrame(stat_table)

    def ets_train_model(self, train_data_all, **kwargs):
        '''
        :param train_data_all: list(iterable obj) that each item is a tuple of (uid, train_data). For now, train_data is 1d array
        :param kwargs:
        :return: etsfit_list: list of tuple(uid, etsfit_object)
        '''
        verbose = 1
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(train_data_all), desc='fitting ets model', unit='ts', leave=True, ascii=True, ncols=105)
        etsfit_list = list()
        for uid, train_data in train_data_all:
            if verbose:
                progress_bar.update(1)

            target_etsfits = self.generate_etsfits(train_data)

            etsfit_list.append((uid, target_etsfits))

        if verbose:
            progress_bar.close()

        return etsfit_list

    def ets_make_forecast(self, etsfit_list, **kwargs):
        '''
        :param etsfit_list:
        :param kwargs:
        :return: uid_set: a list of uids of size N corresponding to the forecast result
                 prediction_set: a 3d array of shape (N, horizon, 1)
        '''
        verbose = 1
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        if verbose:
            progress_bar = tqdm(total=len(etsfit_list), desc='fitting ets model', unit='ts', leave=True, ascii=True,
                                ncols=105)
        uid_set, prediction_set = [], []
        for one_item in etsfit_list:
            if verbose:
                progress_bar.update(1)

            uid = one_item[0]
            target_etsfits = one_item[1]

            num_target = len(target_etsfits)
            prediction = np.zeros((self.sample_horizon, num_target))
            for i in range(num_target):
                # there is no need to use value transformer again, it is already implemented in the ets forecaster
                prediction[:, i] = target_etsfits[i].forecast(self.sample_horizon)

            uid_set.append(uid)
            prediction_set.append(np.expand_dims(prediction, axis=0))

        if verbose:
            progress_bar.close()

        prediction_set = sample_concatenate(prediction_set)
        return uid_set, prediction_set

    def _initialize_sample_generator(self, etsfit_list, **kwargs):
        verbose = 1
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        print('sample generation config: external regressors: {}; dummy hybrid: {}'.format(self.use_ext_regressor, self.dummy_hybrid))

        self.hybrid_sample_generators = list()
        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(etsfit_list), desc='initializing cnn sample generators',
                                unit='ts', leave=True, ascii=True, ncols=105)
        for one_item in etsfit_list:
            if progress_bar is not None:
                progress_bar.update(1)

            pd_regressor = one_item[2]
            regressor, last_season_idx = None, None
            if pd_regressor is not None and self.use_ext_regressor:
                pd_regressor.fillna(0, inplace=True)  ## need to check why??
                if 'season_idx' in pd_regressor.columns:
                    last_season_idx = pd_regressor['season_idx'].values[-1]

                # pd_regressor.drop(columns='season_idx', inplace=True)
                regressor = pd_regressor.values
            # pdb.set_trace()
            ets_nn_gen = ETSNNSampleGenerator(
                one_item[1], self.sample_horizon, self.sample_hislen, self.season_period,
                self.base_season_policy, self.long_season_policy, regressor, last_season_idx
            )

            if self.dummy_hybrid:
                ets_nn_gen.reset_etsfit_dummy()

            self.hybrid_sample_generators.append((one_item[0], ets_nn_gen))

        if progress_bar is not None:
            progress_bar.close()

    def generate_training_samples(self, etsfit_list, holdout_length=0, train_stride=None, num_val=0,
                                  num_train=None, train_random=True, **kwargs):
        verbose = 1
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        if self.hybrid_sample_generators is None:
            self._initialize_sample_generator(etsfit_list, **kwargs)

        x_train_set, y_train_set, x_val_set, y_val_set = [], [], [], []
        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(self.hybrid_sample_generators), desc='extracting nn training/(val) samples',
                                unit='ts', leave=True, ascii=True, ncols=105)
        for uid, ets_nn_gen in self.hybrid_sample_generators:
            if progress_bar is not None:
                progress_bar.update(1)

            ets_nn_gen.set_sample_hislen(self.sample_hislen)
            x_train, y_train, x_val, y_val = \
                ets_nn_gen.create_train_val_set(holdout_length, train_stride, num_val, num_train, train_random)

            if len(x_train) > 0:
                x_train_set.append(x_train)
            if len(y_train) > 0:
                y_train_set.append(y_train)
            if len(x_val) > 0:
                x_val_set.append(x_val)
            if len(y_val) > 0:
                y_val_set.append(y_val)

        if progress_bar is not None:
            progress_bar.close()

        x_train_set = sample_concatenate(x_train_set)
        y_train_set = sample_concatenate(y_train_set)
        x_val_set = sample_concatenate(x_val_set)
        y_val_set = sample_concatenate(y_val_set)

        return x_train_set, y_train_set, x_val_set, y_val_set

    def normalize_nn(self, x_train, y_train, x_val=None, y_val=None,
                     outlier_colidx_x=None, outlier_colidx_y=None, threshold=None, method='neutral'):

        self.normalizer = HybridModelNormalizer(x_train, y_train, outlier_colidx_x, outlier_colidx_y, threshold, method)

        x_train_norm, y_train_norm = self.normalizer.normalize_xy_pair((x_train, y_train))

        # normalize validation set if there is any
        x_val_norm, y_val_norm = None, None
        if x_val is not None and y_val is not None:
            x_val_norm, y_val_norm = self.normalizer.normalize_xy_pair((x_val, y_val))

        return x_train_norm, y_train_norm, x_val_norm, y_val_norm

    def train_nn_model(self, x_train, y_train, x_val=None, y_val=None, num_filters=32, num_layers=2,
                       dropout_rate=(0.1, 0.3), loss='mae', learning_rate=1e-3, optimizer='adam',
                       batch_size=None, epochs=60, use_callback=False, best_model_temp=None, verbose=0, **kwargs):
        input_shape = x_train.shape[1:]
        output_shape = y_train.shape[1:]

        if isinstance(dropout_rate, Number):
            dropout_rate = tuple(np.repeat(dropout_rate, 2))

        self.cnn_model = HybridModelForecaster()
        self.cnn_model.build_model(input_shape, output_shape, num_filters, num_layers, dropout_rate, loss, learning_rate, optimizer)
        self.cnn_model.train_model(x_train, y_train, x_val, y_val, batch_size, epochs, use_callback, best_model_temp, verbose)
        # save cnn model

    def serialize(self, name_prefix):
        pickle.dump(self.normalizer, open(name_prefix + '_normalizer.pkl', 'wb'), protocol=4)
        self.cnn_model.store_model(name_prefix + '_nnmodel')

    def deserialize(self, name_prefix):
        self.normalizer = pickle.load(open(name_prefix + '_normalizer.pkl', 'rb'))
        self.cnn_model = HybridModelForecaster.from_model_file(name_prefix + '_nnmodel')

    def get_forecast_input(self, etsfit_list, holdout_length=0, **kwargs):
        verbose = 1
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        if self.hybrid_sample_generators is None:
            self._initialize_sample_generator(etsfit_list, **kwargs)

        id_set, x_forecast_set, y_ets_normalizer_set, value_transformer_set = [], [], [], []
        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(self.hybrid_sample_generators), desc='generate forecast input x samples',
                                unit='ts', leave=True, ascii=True, ncols=105)
        for uid, ets_nn_gen in self.hybrid_sample_generators:
            if progress_bar is not None:
                progress_bar.update(1)

            id_set.append(uid)

            ets_nn_gen.set_sample_hislen(self.sample_hislen)
            x_forecast, y_ets_normalizer = ets_nn_gen.generate_one_input_sample(holdout_length)

            x_forecast_set.append(np.expand_dims(x_forecast, axis=0))
            y_ets_normalizer_set.append(np.expand_dims(y_ets_normalizer, axis=0))

            target_transformers = [etsfit.value_transformer for etsfit in ets_nn_gen.target_etsfits]
            value_transformer_set.append(target_transformers)

        if progress_bar is not None:
            progress_bar.close()

        id_set = np.array(id_set)
        x_forecast_set = sample_concatenate(x_forecast_set)
        y_ets_normalizer_set = sample_concatenate(y_ets_normalizer_set)
        value_transformer_set = np.array(value_transformer_set)

        return id_set, x_forecast_set, y_ets_normalizer_set, value_transformer_set

    def get_actual_output(self, etsfit_list, holdout_length, **kwargs):
        verbose = 1
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        if self.hybrid_sample_generators is None:
            self._initialize_sample_generator(etsfit_list, **kwargs)

        id_set, y_actual_set, y_ets_scaled_set = [], [], []
        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(self.hybrid_sample_generators), desc='extracting actual y horizon samples',
                                unit='ts', leave=True, ascii=True, ncols=105)
        for uid, ets_nn_gen in self.hybrid_sample_generators:
            if progress_bar is not None:
                progress_bar.update(1)

            id_set.append(uid)

            ets_nn_gen.set_sample_hislen(self.sample_hislen)
            y_actual, y_ets_scaled = ets_nn_gen.extract_one_actual_output_sample(holdout_length)

            y_actual_set.append(np.expand_dims(y_actual, axis=0))
            y_ets_scaled_set.append(np.expand_dims(y_ets_scaled, axis=0))

        if progress_bar is not None:
            progress_bar.close()

        id_set = np.array(id_set)
        y_actual_set = sample_concatenate(y_actual_set)
        y_ets_scaled_set = sample_concatenate(y_ets_scaled_set)

        return id_set, y_actual_set, y_ets_scaled_set

    def hybrid_make_forecast(self, x_forecast_set, y_ets_normalizer_set=None, value_transformer_set=None, **kwargs):
        # using nn normalizer to scale input_x
        nn_normalizer = self.normalizer
        x_forecast_set, _ = nn_normalizer.normalize_one_matrix(
            x_forecast_set, nn_normalizer.scaler_x, nn_normalizer.outlier_colidx_x,
            nn_normalizer.threshold, nn_normalizer.method
        )

        y_prediction = self.cnn_model.predict(x_forecast_set, **kwargs)

        # apply 3-step actual result restoration
        # rescale y output using NN scalers
        y_prediction = nn_normalizer.rescale_one_matrix(y_prediction, nn_normalizer.scaler_y)
        # restore y value using ets normalizers
        y_prediction *= y_ets_normalizer_set
        # final restoration using value transformers
        size_prediction = y_prediction.shape[0]
        size_target = y_prediction.shape[2]
        for i in range(size_prediction):
            for j in range(size_target):
                y_prediction[i, :, j] = value_transformer_set[i][j].inverse(y_prediction[i, :, j])

        return y_prediction

    def compute_smape(self, y_prediction, y_actual, verbose=1, **kwargs):
        num_ts = len(y_prediction)
        num_target = y_prediction.shape[-1]

        if 'id_prediction' not in kwargs.keys() or 'id_actual' not in kwargs.keys():
            warnings.warn('predictiong/actual value id is missing! they might not be aligned!')
            id_prediction = np.arange(num_ts)
            id_actual = np.arange(num_ts)
        else:
            id_prediction = kwargs['id_prediction']
            id_actual = np.array(kwargs['id_actual'])

        smape_table = list()
        for i in range(num_ts):
            cur_id = id_prediction[i]
            match_idx = np.argwhere(id_actual == cur_id)

            one_ts_res = [cur_id]
            for _ in range(2 * num_target):
                one_ts_res.append(np.nan)

            if len(match_idx) > 0:
                match_idx = match_idx[0][0]
                for j in range(num_target):
                    one_ts_res[2 * j + 1] = get_smape_error(y_prediction[i, :, j], y_actual[match_idx, :, j])
                    one_ts_res[2 * j + 2] = np.mean(y_actual[match_idx, :, j])

            smape_table.append(one_ts_res)

        column_names = ['uid']
        for j in range(num_target):
            column_names.append('smape{}'.format(j))
            column_names.append('scale{}'.format(j))
        smape_table = pd.DataFrame(smape_table, columns=column_names)

        if verbose:
            percentiles = list(range(0, 110, 10))
            print('smape percentiles {0}:'.format(percentiles))
            for j in range(num_target):
                smape_value_list = smape_table.iloc[:, 2 * j + 1]
                print('{0} ts with mean smape: {1}'.format(len(smape_value_list), np.mean(smape_value_list)))
                show_value_quantiles(smape_value_list, percentiles)

        return smape_table


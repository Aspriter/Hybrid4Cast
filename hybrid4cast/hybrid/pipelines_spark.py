import numpy as np

from . import HybridPipeline

from ..classicts.ets_forecaster import ETSForecaster
from ..cnn.hybrid_model_forecaster import HybridModelForecaster
from ..hybrid.hybrid_samples import ETSNNSampleGenerator
from ..hybrid.normalizer import HybridModelNormalizer
from ..utils import sample_concatenate, get_smape_error, show_value_quantiles


class HybridModelPipelineSpark(HybridPipeline):

    def __init__(self, sample_horizon=35, sample_hislen=140, season_period=7, use_transform='log', damped_ets=True,
                 base_season_policy=None, long_season_threshold=None, long_season_policy=None, use_ext_regr=False):
        super(HybridModelPipelineSpark, self).__init__(
            sample_horizon, sample_hislen, season_period, use_transform, damped_ets,
            base_season_policy, long_season_threshold, long_season_policy, use_ext_regr
        )

    @staticmethod
    def get_etsfit_statz(etsfit_rdd):

        def mapper_fetch_etsfit(etsfit_data):
            uid, target_etsfits, regressor_mtx, error_log = etsfit_data
            one_row = [uid]
            for etsfit in target_etsfits:
                values = np.expm1(etsfit.proc_ts.values)
                one_row = np.concatenate([one_row, [len(values), np.mean(values), np.std(values)]])
                one_row = np.concatenate([one_row, etsfit.best_config[:-1]])

            return one_row

        etsfit_statz_rdd = etsfit_rdd.map(mapper_fetch_etsfit)
        return etsfit_statz_rdd

    def ets_train_model(self, input_rdd, input_formatter=None, **kwargs):
        '''
        :param input_rdd: RDD data with each item as a tuple (uid, ts_matrix)
        :param input_formatter: format the training data so that ts_matrix has shape (history_len, num_targets)
        :param kwargs:
        :return:
        '''
        def mapper_fit_ets(data):
            uid, ts_matrix, regressor_mtx = data
            target_etsfits, error_log = [], []
            try:
                target_etsfits = self.generate_etsfits(ts_matrix)

            except Exception as e:
                # if no ets model can be fitted, do sth here
                # origin_str = ''
                # for i in range(ts_matrix.shape[1]):
                #     origin_str += str(ts_matrix[:, i]) + '\n'
                # raise Exception('uid: {}; ts index: {}; hislen: {}\n err msg: {}\n error data: {}\n data in etsfit: {}\n alldata: {}'.
                #                 format(uid, i, e, etsfit.proc_ts.values, origin_str))
                error_log.append(ts_matrix)
                error_log.append(e)
                pass

            return uid, target_etsfits, regressor_mtx, error_log

        if input_formatter is not None:
            input_rdd = input_rdd.map(input_formatter)
        etsfit_rdd = input_rdd.map(mapper_fit_ets)

        return etsfit_rdd

    def ets_make_forecast(self, etsfit_rdd, **kwargs):
        '''
        :param etsfit_rdd:
        :param kwargs:
        :return: prediction_rdd: An RDD structure that each item as tuple (uid, prediction_mtx), where prediction_mtx
        has shape (horizon, num_targets)
        '''
        def mapper_ets_forecast(etsfit_tuple, horizon):
            uid, etsfit, regressor_mtx, error_info = etsfit_tuple

            num_target = len(etsfit)
            prediction = np.zeros((horizon, num_target))
            for i in range(num_target):
                prediction[:, i] = etsfit[i].forecast(horizon)

            return uid, prediction

        prediction_rdd = etsfit_rdd.map(lambda x: mapper_ets_forecast(x, self.sample_horizon))

        return prediction_rdd

    def mapper_get_hybrid_training_samples_dp(self, ets_data, holdout_length=0, train_stride=None, num_val=1,
                                           num_train=None, train_random=True):
        uid, target_etsfits, regressor_mtx, error_info = ets_data

        regressor, last_season_idx = None, None
        if regressor_mtx is not None:
            regressor = regressor_mtx.values
            if 'season_idx' in regressor_mtx.columns:
                last_season_idx = regressor_mtx['season_idx'][-1]

        ets_nn_gen = ETSNNSampleGenerator(
            target_etsfits, self.sample_horizon, self.sample_hislen, self.season_period,
            self.base_season_policy, self.long_season_policy, regressor, last_season_idx
        )

        return ets_nn_gen.create_train_val_set(holdout_length, train_stride, num_val, num_train, train_random)

    def _initialize_hybrid_sample_generator(self, etsfit_tuple):
        uid, target_etsfits, regressor_mtx, error_info = etsfit_tuple

        regressor, last_season_idx = None, None
        if regressor_mtx is not None and self.use_ext_regressor:
            regressor = regressor_mtx.values
            if 'season_idx' in regressor_mtx.columns:
                last_season_idx = regressor_mtx['season_idx'].values[-1]

        ets_nn_gen = ETSNNSampleGenerator(
            target_etsfits, self.sample_horizon, self.sample_hislen, self.season_period,
            self.base_season_policy, self.long_season_policy, regressor, last_season_idx
        )

        if self.dummy_hybrid:
            ets_nn_gen.reset_etsfit_dummy()

        return ets_nn_gen

    def generate_training_samples(self, etsfit_rdd, holdout_length=0, train_stride=None, num_val=1,
                                  num_train=None, train_random=True, **kwargs):
        # collect samples per ts from all the time series

        def mapper_get_hybrid_training_samples(ets_data):
            ets_nn_gen = self._initialize_hybrid_sample_generator(ets_data)
            return ets_nn_gen.create_train_val_set(holdout_length, train_stride, num_val, num_train, train_random)

        sample_list = etsfit_rdd.map(mapper_get_hybrid_training_samples).collect()

        x_train_set, y_train_set, x_val_set, y_val_set = [], [], [], []
        for one_ts_sample in sample_list:
            if len(one_ts_sample[0]) > 0:
                x_train_set.append(one_ts_sample[0])
            if len(one_ts_sample[1]) > 0:
                y_train_set.append(one_ts_sample[1])
            if len(one_ts_sample[2]) > 0:
                x_val_set.append(one_ts_sample[2])
            if len(one_ts_sample[3]) > 0:
                y_val_set.append(one_ts_sample[3])

        x_train_set = sample_concatenate(x_train_set)
        y_train_set = sample_concatenate(y_train_set)
        x_val_set = sample_concatenate(x_val_set)
        y_val_set = sample_concatenate(y_val_set)

        return x_train_set, y_train_set, x_val_set, y_val_set

    def normalize_nn(self, x_train, y_train, x_val=None, y_val=None,
                     outlier_colidx_x=None, outlier_colidx_y=None, threshold=None, method='neutral'):
        # in spark implementation, this should be using RDD/DataFrame data structure, here just local version
        self.normalizer = HybridModelNormalizer(x_train, y_train, outlier_colidx_x, outlier_colidx_y, threshold, method)

        x_train_norm, y_train_norm = self.normalizer.normalize_xy_pair((x_train, y_train))

        # normalize validation set if there is any
        x_val_norm, y_val_norm = None, None
        if x_val is not None and y_val is not None:
            x_val_norm, y_val_norm = self.normalizer.normalize_xy_pair((x_val, y_val))

        return x_train_norm, y_train_norm, x_val_norm, y_val_norm

    def train_nn_model(self, x_train, y_train, x_val=None, y_val=None, num_filters=32, num_layers=2,
                       dropout_rate=(0.1, 0.3), loss='mae', learning_rate=1e-3, optimizer='adam',
                       batch_size=None, epochs=50, use_callback=False, best_model_temp=None, verbose=0, **kwargs):
        # this should also be a distributed implementation, but local for now
        input_shape = x_train.shape[1:]
        output_shape = y_train.shape[1:]

        self.cnn_model = HybridModelForecaster()
        self.cnn_model.build_model(input_shape, output_shape, num_filters, num_layers, dropout_rate, loss, learning_rate, optimizer)
        self.cnn_model.train_model(x_train, y_train, x_val, y_val, batch_size, epochs, use_callback, best_model_temp, verbose)
        # save cnn model

    def serialize(self, name_prefix):
        pass

    def deserialize(self, name_prefix):
        pass

    '''
    Below are methods to prepare forecast inputs
    '''
    def mapper_get_hybrid_forecast_input_dp(self, ets_data, holdout_length=0):
        uid, target_etsfits, ts_info = ets_data

        ets_nn_gen = ETSNNSampleGenerator(target_etsfits, self.sample_horizon, self.sample_hislen, self.season_period,
                                          self.base_season_policy, self.long_season_policy)

        x_forecast, y_ets_normalizer = ets_nn_gen.generate_one_input_sample(holdout_length)

        value_transformers = [etsfit.value_transformer for etsfit in target_etsfits]

        return uid, x_forecast, y_ets_normalizer, value_transformers

    def get_forecast_input(self, etsfit_rdd, holdout_length=0):
        # collect samples per ts from all the time series
        def mapper_get_hybrid_forecast_input(etsfit_tuple):
            uid, target_etsfits, regressor_mtx, error_info = etsfit_tuple

            ets_nn_gen = self._initialize_hybrid_sample_generator(etsfit_tuple)
            x_forecast, y_ets_normalizer = ets_nn_gen.generate_one_input_sample(holdout_length)

            value_transformers = [etsfit.value_transformer for etsfit in target_etsfits]

            return uid, x_forecast, y_ets_normalizer, value_transformers

        input_list = etsfit_rdd.map(mapper_get_hybrid_forecast_input).collect()

        id_set, x_input_set, y_ets_normalizer_set, value_transformer_set = [], [], [], []
        for one_input in input_list:
            id_set.append(one_input[0])
            x_input_set.append(np.expand_dims(one_input[1], axis=0))
            y_ets_normalizer_set.append(np.expand_dims(one_input[2], axis=0))
            value_transformer_set.append(one_input[3])

        id_set = np.array(id_set)
        x_input_set = sample_concatenate(x_input_set)
        y_ets_normalizer_set = sample_concatenate(y_ets_normalizer_set)

        return id_set, x_input_set, y_ets_normalizer_set, value_transformer_set

    def get_actual_output(self, etsfit_list, holdout_length, **kwargs):
        pass

    def hybrid_make_forecast(self, x_forecast, y_ets_normalizer=None, y_value_transformer=None, **kwargs):
        '''
        this method is a local CNN forecast implementation at the driver node batch by batch, using the corresponding
        method provided by HybridModelForecaster(local version). The forecaster will generate boot-straping alike results
        contains num_samples of predictions.
        BE CAREFUL TO RESTORE THE ACTUAL OUTPUT USING TWO PHASES:
        1. use NN_normalizer to rescale;
        2. finally to use ets_normalizer(per ts specific) to restore the actual value
        :return: an rdd output of format (uid, y_prediction)
        '''
        forecast_size = len(x_forecast)
        id_set, spark_context = list(range(forecast_size)), None
        if 'id_set' in kwargs.keys():
            id_set = kwargs['id_set']
            kwargs.pop('id_set', None)
        if 'sc' in kwargs.keys():
            spark_context = kwargs['sc']
            kwargs.pop('sc', None)

        # using nn normalizer to scale input_x
        nn_normalizer = self.normalizer
        x_forecast, _ = nn_normalizer.normalize_one_matrix(
            x_forecast, nn_normalizer.scaler_x, nn_normalizer.outlier_colidx_x
        )

        # do forecast (local version)
        y_prediction = self.cnn_model.predict(x_forecast, **kwargs)

        # apply 3-step actual result restoration
        y_prediction = nn_normalizer.rescale_one_matrix(y_prediction, nn_normalizer.scaler_y)
        y_prediction *= y_ets_normalizer
        size_prediction = y_prediction.shape[0]
        size_target = y_prediction.shape[2]
        for i in range(size_prediction):
            for j in range(size_target):
                y_prediction[i, :, j] = y_value_transformer[i][j].inverse(y_prediction[i, :, j])

        # convert the collected list output back to rdd format
        prediction_holder = []
        for i in range(len(id_set)):
            prediction_holder.append((id_set[i], y_prediction[i, :, :]))
        prediction_rdd = spark_context.parallelize(prediction_holder)

        return prediction_rdd

    @staticmethod
    def compute_smape(prediction_rdd, actual_rdd, verbose=1, **kwargs):

        def mapper_compute_smape(hybrid_output):
            uid, (prediction, actual) = hybrid_output
            num_target = actual.shape[1]

            row_ret = [uid]
            for i in range(2 * num_target):
                row_ret.append(np.nan)

            if prediction.shape == actual.shape:
                for i in range(num_target):
                    row_ret[i * 2 + 1] = get_smape_error(prediction[:, i], actual[:, i])
                    row_ret[i * 2 + 2] = np.mean(actual[:, i])

            return tuple(row_ret)

        error_rdd = prediction_rdd.join(actual_rdd).map(mapper_compute_smape)

        if verbose:
            HybridModelPipelineSpark.show_smape_rdd(error_rdd, **kwargs)

        return error_rdd

    @staticmethod
    def show_smape_rdd(smape_rdd, m4_type=None):
        first_row = smape_rdd.first()
        num_target = int(len(first_row)/2)

        percentiles = list(range(0, 110, 10))
        print('smape percentiles {0}:'.format(percentiles))
        target_smape = list()
        for i in range(num_target):
            one_target_smape_list = smape_rdd.map(lambda x: x[2 * i + 1]).collect()
            tar_smape = np.nanmean(one_target_smape_list)
            target_smape.append(tar_smape)
            print('{0} {1} ts with mean smape: {2}'.format(len(one_target_smape_list), m4_type, tar_smape))
            show_value_quantiles(one_target_smape_list, percentiles)

        return target_smape


'''
    draft for weekly training logic for ETS-CNN in spark
    cpg_all_data: list of TS data matrix

    1. train, val = []
    for each TS:
        generate train_per_cpg, val_per_cpg using ETSNNSampleGenerator
        ETSNNSampleGenerator is actually a functional wrapper of ETSForecaster,
        train.append(train_per_cpg)
        val.append(val_per_cpg)
    #### for each cpg, save per TS's ETSForecaster for input_x reconstruction in daily forecast

    2. use train, val to fit NN model, using HybridModelForecaster
    #### save model json, h5 and normalizers for forecasting
'''

'''
    draft for daily forecast logic for ETS-CNN in spark
    input: per TS per target ETSGridSearch
           new delta data since last training
           model serialization result

    1.  load per cpg per target's ETSForecaster, 
        extend ETSForecaster.etsfit_mtx with new data (delta data since last training)
        wrap it with ETSNNSampleGenerator and generate input_x with generate_one_test_sample

    2.  load model file into class HybridModelForecaster, 
        forecast and restore the actual result using input_x & model normalizer
'''

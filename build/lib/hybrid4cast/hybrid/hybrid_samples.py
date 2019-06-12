import warnings
import pandas as pd
import numpy as np
from numbers import Number

from ..classicts.ts_helper import TimeSeriesPreprocessor
from ..utils import array_division


class ETSNNSampleGenerator:

    hislen_horizon_ratio = 4
    base_season_policy_options = ['force', 'byets']
    long_season_policy_options = ['shifbase', 'shifspec', 'cyclspec']

    def __init__(self, target_etsfits, sample_horizon, sample_hislen=None, base_season=None, base_season_policy=None,
                 long_season_policy=None, ext_features=None, last_in_season_idx=None):
        if target_etsfits is None or len(target_etsfits) < 1:
            raise Exception('ETS-CNN sample initialization failed, there must be at least one ETS fitted time series')
        for etsfit in target_etsfits:
            if etsfit.etsfit_mtx is None:
                if etsfit.best_model is None:
                    raise Exception('each target TS should have a fitted ets model to generate features')
                else:
                    etsfit.fill_fit_matrix()

        self.target_etsfits = target_etsfits

        self.sample_horizon = sample_horizon
        self.sample_hislen = self.hislen_horizon_ratio * sample_horizon if sample_hislen is None else sample_hislen

        self.ts_length = self.get_etsfit_pooled_length(target_etsfits)
        if isinstance(base_season, Number) and base_season < 2:
            base_season = None

        self.base_season_policy = None
        self.pooled_base_season = None
        self.base_season_sequence = None
        if base_season_policy in self.base_season_policy_options:
            self.base_season_policy = base_season_policy

        if self.base_season_policy is not None:
            if self.base_season_policy == 'force':
                self.pooled_base_season = base_season
            else:
                self.pooled_base_season = self.get_etsfit_pooled_base_season(target_etsfits, base_season)

            self.base_season_sequence = np.zeros(self.ts_length)
            if self.pooled_base_season is not None:
                start_season_idx = 0 if last_in_season_idx is None else last_in_season_idx - self.ts_length
                self.base_season_sequence = np.arange(start_season_idx, start_season_idx + self.ts_length) * 2 * np.pi / self.pooled_base_season

        self.long_season_policy = None
        self.pooled_long_season = None
        self.long_season_acf = None
        self.long_season_sequence = None
        if long_season_policy in self.long_season_policy_options:
            self.long_season_policy = long_season_policy

        if self.long_season_policy is not None:
            if self.long_season_policy.endswith('spec'):
                self.pooled_long_season = self.get_etsfit_pooled_long_season(
                    self.target_etsfits, self.target_etsfits[0].proc_ts.long_season_threshold
                )
            else:
                self.pooled_long_season = base_season

            if self.long_season_policy.startswith('shif'):
                acf_values = list()
                for etsfit in self.target_etsfits:
                    acf_values.append(etsfit.proc_ts.get_long_season_acf(self.pooled_long_season))
                self.long_season_acf = np.mean(acf_values)
            else:
                self.long_season_sequence = np.zeros(self.ts_length)
                if self.pooled_long_season is not None:
                    self.long_season_sequence = np.arange(self.ts_length) * 2 * np.pi / int(self.pooled_long_season)

        self.ext_features = ext_features

    def set_sample_hislen(self, sample_hislen):
        self.sample_hislen = sample_hislen

    # reset the fitted ets values as dummy values to do none-hybridization experiments
    def reset_etsfit_dummy(self):
        for etsfit in self.target_etsfits:
            etsfit.fill_dummy_fitmtx()

    @staticmethod
    def get_etsfit_pooled_length(etsfit_list):
        pooled_length = 0
        for etsfit in etsfit_list:
            if etsfit.proc_ts.length > pooled_length:
                pooled_length = etsfit.proc_ts.length
        return pooled_length

    @staticmethod
    def get_etsfit_pooled_period_table(etsfit_list):
        pooled_period_table = pd.DataFrame()
        for etsfit in etsfit_list:
            if etsfit.proc_ts.sig_period_table is not None:
                pooled_period_table = pooled_period_table.append(etsfit.proc_ts.sig_period_table)

        if len(pooled_period_table) == 0:
            pooled_period_table = None

        return pooled_period_table

    @staticmethod
    def get_etsfit_pooled_base_season(etsfit_list, base_season=None):
        return TimeSeriesPreprocessor.get_dominant_base_season(
            ETSNNSampleGenerator.get_etsfit_pooled_period_table(etsfit_list), base_season
        )

    @staticmethod
    def get_etsfit_pooled_long_season(etsfit_list, long_season_threshold=None):
        return TimeSeriesPreprocessor.get_most_significant_long_season(
            ETSNNSampleGenerator.get_etsfit_pooled_period_table(etsfit_list), long_season_threshold
        )

    @staticmethod
    def get_etsfit_pooled_season_component(etsfit_list):
        season_component = None
        for etsfit in etsfit_list:
            this_seacomp = etsfit.get_fitted_season_component()
            if this_seacomp is not None:
                season_component = this_seacomp
        return season_component

    '''
    go through the entire time series history to extract training(and validation) samples 
    '''
    def create_train_val_set(self, holdout_length=0, train_stride=None, num_val=1, num_train=None, train_random=True):
        num_train = 1e4 if num_train is None else num_train
        train_stride = self.sample_horizon if train_stride is None else int(train_stride)

        train_set_x, train_set_y, val_set_x, val_set_y = [], [], [], []

        # sample_pivot_index is the location where divides the sample as input history window and output forecast horizon, namely:
        # [sample_pivot_index - sample_hislen : sample_pivot_index]    is input history window
        # [sample_pivot_index : sample_pivot_index + sample_horizon]   is output forecast horizon
        sample_pivot_index = self.ts_length - holdout_length - self.sample_horizon
        while sample_pivot_index >= self.sample_hislen:
            # for each sample, first create input x, then output y
            sample_input_x = self._generate_input_x(sample_pivot_index)
            sample_output_y = self._generate_train_output_y(sample_pivot_index)
            if num_val > 0:
                val_set_x.append(sample_input_x)
                val_set_y.append(sample_output_y)
                num_val -= 1
            else:
                train_set_x.append(sample_input_x)
                train_set_y.append(sample_output_y)
                num_train -= 1

            this_stride = train_stride
            if train_random:
                this_stride = np.random.randint(int(train_stride / 2), int(train_stride * 3 / 2))
            sample_pivot_index -= this_stride

            if num_val == 0 and num_train == 0:
                break

        return np.array(train_set_x), np.array(train_set_y), np.array(val_set_x), np.array(val_set_y)

    '''
    to generate one input x sample and ets normalizer (forecasted ets seasonal levels) to recover the result, pivot_index
    is the end of index_window_end + 1. The generated input x should have length self.sample_hislen, if pivot_index is too
    small, 0 values will be padded at the top of input sample
    '''
    def generate_one_input_sample(self, holdout_length=0):
        pivot_index = self.ts_length - holdout_length

        test_input_x = self._generate_input_x(pivot_index)
        # if the total TS length is too short to fill one input sample (sample_hislen * feature_cols), pad 0
        sample_shape = test_input_x.shape
        if sample_shape[0] < self.sample_hislen:
            test_input_x = np.concatenate([np.zeros((self.sample_hislen - sample_shape[0], sample_shape[1])), test_input_x], axis=0)

        num_target = len(self.target_etsfits)
        test_y_ets_normalizer = np.zeros((self.sample_horizon, num_target))
        for i in range(num_target):
            test_y_ets_normalizer[:, i] = self.target_etsfits[i].get_horizon_seasonal_levels(self.sample_horizon, pivot_index)

        return test_input_x, test_y_ets_normalizer

    def extract_one_actual_output_sample(self, holdout_length):
        pivot_index = self.ts_length - holdout_length

        num_target = len(self.target_etsfits)
        actual_y = np.zeros((self.sample_horizon, num_target))
        ets_scaled_y = actual_y.copy()
        for i in range(num_target):
            actual_value = self.target_etsfits[i].etsfit_mtx[pivot_index:pivot_index + self.sample_horizon, 0]
            ets_scaled_value = array_division(
                self.target_etsfits[i].etsfit_mtx[pivot_index:pivot_index + self.sample_horizon, 0],
                self.target_etsfits[i].etsfit_mtx[pivot_index:pivot_index + self.sample_horizon, 3]
            )
            actual_length = len(actual_value)
            if actual_length != self.sample_horizon:
                warnings.warn('expected actual value length {0}; only {1} get'.format(self.sample_horizon, actual_length))
                if actual_length > 0:
                    actual_y[:actual_length, i] = actual_value
                    ets_scaled_y[:actual_length, i] = ets_scaled_value
            else:
                actual_y[:, i] = actual_value
                ets_scaled_y[:, i] = ets_scaled_value

        return actual_y, ets_scaled_y

    '''
    generate the input feature matrix given end_index, which is the same as pivot_index. If end_index is too small, only
    that many rows will be generated
    '''
    def _generate_input_x(self, end_index):
        # generate target values normalized by the ETS seasonal level values
        num_target = len(self.target_etsfits)
        actual_sample_length = self.sample_hislen if end_index > self.sample_hislen else end_index
        input_x = np.zeros((actual_sample_length, num_target))
        for i in range(num_target):
            target_etsmtx = self.target_etsfits[i].etsfit_mtx
            # pdb.set_trace()
            input_x[:, i] = array_division(target_etsmtx[end_index - actual_sample_length:end_index, 0],
                                           target_etsmtx[end_index - actual_sample_length:end_index, 3])

        # adding shifted (normalized) target feature column
        if self.long_season_policy is not None:
            for i in range(num_target):
                etsfit = self.target_etsfits[i]
                shifted_matrix = np.zeros((actual_sample_length, 2))
                if self.long_season_policy.startswith('shif'):
                    if self.pooled_long_season is not None:
                        shifted_value_end_index = end_index - self.pooled_long_season
                        if shifted_value_end_index > 0:
                            shifted_value_start_index = int(np.max([0, shifted_value_end_index - actual_sample_length]))
                            shifted_values = array_division(
                                etsfit.etsfit_mtx[shifted_value_start_index:shifted_value_end_index, 0],
                                etsfit.etsfit_mtx[shifted_value_start_index:shifted_value_end_index, 3]
                            )
                            shifted_value_len = len(shifted_values)
                            if shifted_value_len > 0:
                                shifted_matrix[-shifted_value_len:, 0] = shifted_values
                                shifted_matrix[-shifted_value_len:, 1] = self.long_season_acf
                else:
                    this_sample_season_seq = self.long_season_sequence[end_index - actual_sample_length:end_index]
                    shifted_matrix[:, 0] = np.sin(this_sample_season_seq)
                    shifted_matrix[:, 1] = np.cos(this_sample_season_seq)

                input_x = np.concatenate([input_x, shifted_matrix], axis=1)

        # adding external regressors
        if self.ext_features is not None:
            sample_ext_regr = self.ext_features[end_index - actual_sample_length:end_index].copy()
            input_x = np.concatenate([input_x, sample_ext_regr], axis=1)

        if self.base_season_policy is not None:
            # append cylindrical columns of the base seasonality
            season_cylindrical_cols = np.zeros((actual_sample_length, 2))
            season_component = self.get_etsfit_pooled_season_component(self.target_etsfits)
            # use the dominant base season value in spectrum analysis as indicator
            # season_component = self.get_etsfit_pooled_base_season(self.target_etsfits, self.pooled_base_season)
            if self.base_season_policy == 'force' or season_component is not None:
                this_sample_season_seq = self.base_season_sequence[end_index - actual_sample_length:end_index]
                season_cylindrical_cols[:, 0] = np.sin(this_sample_season_seq)
                season_cylindrical_cols[:, 1] = np.cos(this_sample_season_seq)
            input_x = np.concatenate([input_x, season_cylindrical_cols], axis=1)

        return input_x

    def _generate_train_output_y(self, start_index):
        # generate target values normalized by the ETS seasonal level values in the horizon
        num_target = len(self.target_etsfits)
        output_y = np.zeros((self.sample_horizon, num_target))
        for i in range(num_target):
            target_etsfit = self.target_etsfits[i]
            if start_index + self.sample_horizon > self.ts_length:
                raise Exception('Not enough data to fill into training output y')

            output_y[:, i] = array_division(target_etsfit.etsfit_mtx[start_index:start_index + self.sample_horizon, 0],
                                            target_etsfit.get_horizon_seasonal_levels(self.sample_horizon, start_index))

        return output_y

    '''
    static method to restore actual values from cnn samples
    '''
    @staticmethod
    def restore_actual_from_ets(y_nn_output, y_ets_normalizer):
        if not np.array_equal(y_nn_output.shape, y_ets_normalizer.shape):
            raise Exception('the ets normalizer is incompatible with nn prediction output')

        for i in range(y_nn_output.shape[1]):
            y_nn_output[: i] *= y_ets_normalizer[:, i]

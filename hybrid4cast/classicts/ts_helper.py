import pdb
import numpy as np
import pandas as pd
import warnings
from numbers import Number

from scipy.stats import boxcox, shapiro
from scipy.special import inv_boxcox
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.tsatools import detrend

from . import smooth_autocorr


class BoxCoxTransformer:

    options = ('boxcox', 'log')

    def __init__(self, method='boxcox', param=None):
        self.method = method
        if method not in self.options:
            warnings.warn('Unrecognized transformer type: {0}, no transform is used!'.format(method))
            self.method = None

        self.param = param

    def __str__(self):
        return str((self.method, self.param))

    def convert(self, ts_array):
        if self.method is None:
            return ts_array

        if self.method == 'boxcox':
            p_norm_original = _get_normal_test_p_value(ts_array)
            new_array, best_lambda = _get_boxcox_param(ts_array, p_norm_original)
            if best_lambda is None or best_lambda == 0:
                self.method = 'log'
            else:
                self.param = best_lambda
                return new_array

            new_array = np.log1p(ts_array)
            if _get_normal_test_p_value(new_array) > p_norm_original:
                return new_array
            else:
                self.method = None
                return ts_array

        if self.method == 'log':
            return np.log1p(ts_array)

    def inverse(self, ts_array):
        if self.method is None:
            return ts_array

        if self.method == 'log':
            return np.expm1(ts_array)
        else:
            return inv_boxcox(ts_array, self.param)


def _get_boxcox_param(x, p_norm_before=None):
    if p_norm_before is None:
        p_norm_before = _get_normal_test_p_value(x)

    x_new, x_lambda = boxcox(x)

    # x_lambda = np.round(x_lambda / 0.5)
    x_lambda = np.min([1, x_lambda])
    x_lambda = np.max([0.3, x_lambda])
    # x_lambda *= 0.5

    x_new = boxcox(x, x_lambda)
    if _get_normal_test_p_value(x_new) > p_norm_before:
        return x_new, x_lambda
    else:
        return None, None


class TimeSeriesPreprocessor:

    # parameters for TS stationary test
    adf_regression_type = 'ct'
    adf_critical_threshold = 0.05

    # multipliers for deciding default history length
    hislen_horizon_multiplier = 20
    hislen_long_season_multiplier = 2.2

    # for detecting pitches in spectrum array
    sequence_deviation_threshold = 10
    sequence_smoothing_level = 0.5
    significant_period_finder_lag = 'base_season'

    # for deciding/aligning base/long season periods
    long_season_threshold_horizon_multiplier = 6
    flex_ratio_long_season_threshold = 0.9
    flex_ratio_base_season_deviation = 0.2

    def __init__(self, ts_array, horizon, base_season=None, long_season_threshold=None, transform=None, initialize=True):
        if np.ndim(ts_array) != 1:
            raise ValueError('it must be a 1d array!')

        self.values = ts_array
        self.value_transformer = None
        self.length = len(self.values)
        self.horizon = horizon

        self.trend = None
        self.sig_period_table = None
        self.base_season = base_season
        self.long_season_threshold = long_season_threshold
        self.sig_long_season = None
        if not isinstance(long_season_threshold, Number):
            self.long_season_threshold = self.long_season_threshold_horizon_multiplier * self.horizon

        # self.hislen_horizon_multiplier = hislen_horizon_multiplier
        # self.significant_period_finder_lag = sig_period_lag

        if initialize:
            self.initialize(transform)

    def initialize(self, transform=None):
        self.value_transformer = BoxCoxTransformer(transform)
        self.values = self.value_transformer.convert(self.values)

        if not _is_stationary(self.values, self.adf_regression_type, self.adf_critical_threshold):
            self.trend = 'linear'

        self.sig_period_table = _get_significant_periods(self.values, 'linear', self.significant_period_finder_lag,
                                                         self.sequence_deviation_threshold, self.sequence_smoothing_level)

        self.sig_long_season = self.get_most_significant_long_season(self.sig_period_table, self.long_season_threshold)
        self.length = self.get_optimal_history_length()

    def __str__(self):
        value_dict = dict()
        value_dict['actual_len'] = len(self.values)
        value_dict['optimal_len'] = self.length
        value_dict['value_transform'] = str(self.value_transformer)
        value_dict['trend'] = self.trend
        value_dict['horizon'] = self.horizon

        value_dict['base_season'] = self.base_season
        value_dict['long_season_thres'] = self.long_season_threshold
        value_dict['sig_long_season'] = self.sig_long_season
        value_dict['sig_period_table'] = self.sig_period_table
        return str(value_dict)

    def update_ts_length(self, actual_length=None):
        actual_length = self.length if actual_length is None else actual_length

        self.values = self.values[-actual_length:]

        if not _is_stationary(self.values, self.adf_regression_type, self.adf_critical_threshold):
            self.trend = 'linear'
        else:
            self.trend = None

    @staticmethod
    def get_most_significant_long_season(period_table, long_season_threshold=None):
        if period_table is None:
            return None

        candidate = period_table
        if long_season_threshold is not None:
            candidate = period_table[period_table.freq >=
                                     TimeSeriesPreprocessor.flex_ratio_long_season_threshold * long_season_threshold]
        if len(candidate) == 0:
            return None

        return int(candidate.iloc[np.argmax(candidate.mag.values), 0])

    @staticmethod
    def get_dominant_season(period_table=None):
        if period_table is None:
            return None

        return int(period_table.iloc[np.argmax(period_table.mag.values), 0])

    @staticmethod
    def get_shortest_season(period_table=None):
        if period_table is None:
            return None

        return int(np.min(period_table.freq.values))

    @staticmethod
    def get_dominant_base_season(period_table=None, base_season=None):

        if period_table is None:
            return None
        else:
            if base_season is None:
                # return TimeSeriesPreprocessor.get_dominant_season(period_table)
                return TimeSeriesPreprocessor.get_shortest_season(period_table)
            else:
                candidate = np.abs(period_table.freq - base_season) <= \
                            np.ceil(TimeSeriesPreprocessor.flex_ratio_base_season_deviation * base_season)
                candidate = period_table[candidate]
                if len(candidate) == 0:
                    return None
                else:
                    return base_season

    @staticmethod
    def get_dominant_base_season_v1(period_table=None, base_season=None):

        if base_season is None and period_table is None:
            return None
        elif base_season is None and period_table is not None:
            return TimeSeriesPreprocessor.get_shortest_season(period_table)
            # return TimeSeriesPreprocessor.get_dominant_season(period_table)
        elif base_season is not None and period_table is None:
            return base_season
        else:
            candidate = np.abs(period_table.freq - base_season) <= TimeSeriesPreprocessor.flex_ratio_base_season_deviation * base_season
            candidate = period_table[candidate]
            if len(candidate) == 0:
                return base_season
            else:
                return candidate.iloc[np.argmax(candidate.mag.values), 0]

    def get_optimal_history_length(self):
        hislen = self.hislen_horizon_multiplier * self.horizon
        if self.sig_long_season is not None:
            hislen = np.max([hislen, self.hislen_long_season_multiplier * self.sig_long_season])

        hislen = np.min([hislen, len(self.values)])
        return int(hislen)

    def get_long_season_acf(self, lag=None):
        lag = self.sig_long_season if lag is None else lag
        return smooth_autocorr(self.values, lag, partial=False)


def _is_stationary(x, regression_type='ct', critial_threshold=0.05):
    try:
        max_lag = int(np.floor((len(x)-1)**(1/3)))
        adf_result = adfuller(x, regression=regression_type, maxlag=max_lag)
        return adf_result[1] < critial_threshold
    except:
        return True


def _get_normal_test_p_value(x):
    return shapiro(x)[1]


def _get_significant_periods(ts_array, detrend=None, lag=None, threshold=10, alpha=0.75):
    array_length = len(ts_array)

    if not isinstance(lag, Number):
        lag = np.min([5, array_length / 20])
    lag = int(lag)
    # if the lag span is too big, directly return None
    if lag > 0.9 * array_length / 2:
        return None

    frequences, pgram = periodogram(ts_array, detrend=detrend, scaling='spectrum')

    pgram_rev = pgram[::-1]

    signal, x_filtered, avg_filter, std_filter = _sequence_alarm(pgram_rev, lag, threshold, alpha)

    signal = signal[::-1]
    # set a cutoff long period threshold, equivalent as ignore the first few values since they are unstable
    pgram_skip_num = int(np.max([3, array_length // 400]))
    signal[:pgram_skip_num] = 0
    sig_idx = np.where(signal == 1)[0]
    if len(sig_idx) == 0:
        return None

    # print(pgram[sig_idx])
    sig_periods = [int(np.round(array_length/idx)) for idx in sig_idx]
    sig_table = pd.DataFrame({'freq': sig_periods, 'mag': pgram[sig_idx]})
    sig_table = sig_table.groupby('freq')['mag'].mean()
    sig_table = sig_table.reset_index().rename(columns={'index': 'freq'})

    return sig_table


def _detrend(x, method):
    if method == 'diff':
        return np.diff(x)

    if isinstance(method, str):
        method = dict(constant=0, linear=1, quadratic=2, cubic=3)[method]

    if isinstance(method, Number):
        x = detrend(x, method)

    return x


def _sequence_alarm(x, lag, threshold=3, alpha=0.8):
    signal = np.zeros(len(x))
    x_filtered = x.copy()
    avg_filter = np.repeat(np.nan, len(x))
    std_filter = np.repeat(np.nan, len(x))
    avg_filter[lag - 1] = np.mean(x[:lag])
    std_filter[lag - 1] = np.std(x[:lag])
    for i in range(lag, len(x)):
        this_dev = x[i] - avg_filter[i - 1]
        if np.abs(this_dev) > threshold * std_filter[i - 1]:
            signal[i] = np.sign(this_dev)
            x_filtered[i] = alpha * x[i] + (1 - alpha) * x_filtered[i - 1]
        else:
            x_filtered[i] = x[i]

        avg_filter[i] = np.mean(x_filtered[i - lag:i])
        std_filter[i] = np.std(x_filtered[i - lag:i])

    return signal, x_filtered, avg_filter, std_filter


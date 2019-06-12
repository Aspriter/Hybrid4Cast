import math
import warnings
import numpy as np
import pandas as pd
from numbers import Number

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ..classicts.ts_helper import TimeSeriesPreprocessor, BoxCoxTransformer
from ..utils import parse_int


class ETSForecaster:

    def __init__(self, ts_array=None, horizon=None, season_period=None, long_season_threshold=None,
                 damped=True, use_transform=None, criterion='aicc', initialize=True):
        if criterion not in ['aic', 'bic', 'aicc']:
            raise ValueError('the criterion has to be one of aic, bic or aicc!')
        self.criterion = criterion

        if isinstance(season_period, Number):
            season_period = None if season_period < 2 else season_period

        self.proc_ts = None
        self.value_transformer = None
        self.season_period = None
        self.damped = damped

        self.configs = None

        # metrics used to find/store the best ETS model
        self.best_metric = 1e10
        self.best_config = None
        self.best_model = None
        self.metric_table = None

        # this is a ets feature matrix holder, where stores the fitted ets level/season values of the best model
        # etsfit_mtx[:, 0] : the actual target time series
        # etsfit_mtx[:, 1] : fitted ETS levels
        # etsfit_mtx[:, 2] : fitted ETS seasonal values
        # etsfit_mtx[:, 3] : fitted ETS seasonal-level combination values based on ETS seasonal component
        self.etsfit_mtx = None

        if initialize:
            self.proc_ts = TimeSeriesPreprocessor(ts_array, horizon, season_period, long_season_threshold, use_transform, True)
            self.value_transformer = self.proc_ts.value_transformer
            self.season_period = self.proc_ts.get_dominant_base_season(self.proc_ts.sig_period_table, season_period)

    @classmethod
    def from_r_result_tsv(cls, mtx_str, ets_info_str):
        etsfit = cls(initialize=False)
        etsfit.etsfit_mtx = np.array([pd.Series(one_row.split(",")).apply(float).values for one_row in mtx_str.split(";")])
        etsfit.proc_ts = TimeSeriesPreprocessor(etsfit.etsfit_mtx[:, 0], 0, transform='log', initialize=False)

        ets_info = ets_info_str.split(';')
        season_value_strings = ets_info[0].split(',')
        etsfit.season_period = parse_int(season_value_strings[0])
        etsfit.proc_ts.sig_long_season = parse_int(season_value_strings[1])

        period_table = pd.DataFrame({'freq': [etsfit.season_period, etsfit.proc_ts.sig_long_season], 'mag': [1, 1]})
        etsfit.proc_ts.sig_period_table = period_table.dropna()

        ets_component_str = ets_info[1].split(',')
        season_component = None
        if ets_component_str[2] == 'A':
            season_component = 'add'
        elif ets_component_str[2] == 'M':
            season_component = 'mul'
        etsfit.best_config = [None, season_component, None, False]
        etsfit.value_transformer = BoxCoxTransformer('log')

        # pdb.set_trace()
        return etsfit

    def update_ts_length(self, actual_length=None):
        self.proc_ts.update_ts_length(actual_length)

    def get_config_list(self):
        trend_params = {None}
        damped_params = {False}
        if self.proc_ts.trend is not None:
            trend_params.add('add')
            if self.damped:
                damped_params.add(True)
        season_period_params = {None}
        season_period_params.add(self.season_period)
        season_params = [None]
        if self.season_period is not None and self.season_period != 1:
            season_params = ['add', 'mul', None]

        configs = list()
        for t in trend_params:
            for s in season_params:
                for sp in season_period_params:
                    for d in damped_params:
                        if sp is None and s is not None:
                            continue
                        if sp is not None and s is None:
                            continue
                        # only try add/damped trend comb
                        if t is not 'add' and d:
                            continue

                        configs.append([t, s, sp, d])

        return configs

    '''
    remember to suppress warnings since grid search is noisy
    '''
    def fit_grid_search(self):
        self.configs = self.get_config_list()
        metric_table = []
        for cfg in self.configs:
            t, s, sp, d = cfg

            ets_fit = self._fit_one_config(self.proc_ts.values, t, s, sp, d)
            if ets_fit is None:
                continue

            cur_metric = getattr(ets_fit, self.criterion)
            # cfg.append(False)
            cfg.append(cur_metric)
            metric_table.append(cfg)
            # print(cfg)

            # for aic, bic, aicc criterion, the smaller the better
            if cur_metric < self.best_metric:
                self.best_metric = cur_metric
                self.best_config = cfg
                self.best_model = ets_fit

        if len(metric_table) == 0:
            raise Exception('no ets model can be fitted for this time series')

        # try damped is True for the best fit model
        # if self.damped:
        #     t, s, sp, d, metric = self.best_config
        #     ets_fit = self._fit_one_config(self.proc_ts.values, t, s, sp, True)
        #     if ets_fit is not None:
        #         cur_metric = getattr(ets_fit, self.criterion)
        #         this_config = [t, s, sp, True, cur_metric]
        #         metric_table.append(this_config)
        #         if cur_metric < self.best_metric:
        #             self.best_metric = cur_metric
        #             self.best_config = this_config
        #             self.best_model = ets_fit

        # there is no need to fill the ets_level_season_matrix, do it only when generating cnn samples
        self.metric_table = pd.DataFrame(metric_table, columns=['trend', 'season', 'seaprd', 'damped', self.criterion])
        # print(self.metric_table) # print the metric table for debug

    @staticmethod
    def _fit_one_config(ts_array, trend, seasonal, seasonal_period, damped):
        try:
            ets_model = ExponentialSmoothing(
                ts_array, trend=trend, damped=damped,
                seasonal=seasonal, seasonal_periods=seasonal_period
            )
            return ets_model.fit(optimized=True)
        except NotImplementedError:
            return None
        except FloatingPointError:
            return None

    def fill_fit_matrix(self, etsfit=None):
        etsfit = self.best_model if etsfit is None else etsfit
        ets_model = etsfit.model
        ts_len = len(ets_model.endog)
        fit_matrix = np.zeros((ts_len, 4))
        fit_matrix[:, 0] = ets_model.endog
        fit_matrix[:, 1] = etsfit.level
        fit_matrix[:, 2] = etsfit.season

        level_shifted = np.zeros(ts_len)
        if np.isfinite(ets_model.params['initial_level']):
            level_shifted[0] = ets_model.params['initial_level']
        level_shifted[1:] = etsfit.level[:-1]

        if ets_model.seasonal is not None:
            season_shifted = np.zeros(ts_len)
            season_shifted[0:ets_model.seasonal_periods] = ets_model.params['initial_seasons']
            season_shifted[ets_model.seasonal_periods:] = etsfit.season[:-ets_model.seasonal_periods]
            if ets_model.seasonal == 'add':
                level_shifted += season_shifted
            elif ets_model.seasonal == 'mul':
                level_shifted *= season_shifted

        fit_matrix[:, 3] = level_shifted
        self.etsfit_mtx = fit_matrix

    def fill_dummy_fitmtx(self):
        if self.best_model is None:
            raise Exception('Dummy fit matrix can only filled with a fitted ets model!')

        ets_model = self.best_model.model
        ts_len = len(ets_model.endog)
        fit_matrix = np.zeros((ts_len, 4))
        fit_matrix[:, 0] = ets_model.endog
        fit_matrix[:, 1] = 1
        fit_matrix[:, 3] = 1

        season_component = self.get_fitted_season_component()
        if self.season_period is not None and season_component is not None:
            if season_component == 'mul':
                fit_matrix[:, 2] = 1

        self.etsfit_mtx = fit_matrix

    # make forecast using ets alone, warnings can be thrown when there is no time index
    def forecast(self, horizon):
        if self.best_model is None:
            raise Exception('an ets model has to be fitted first!')

        result = self.best_model.forecast(horizon)
        result = self.value_transformer.inverse(result)

        invalid_idx = np.where(~np.isfinite(result))[0]
        if len(invalid_idx) > 0:
            warnings.warn('there are {0} invalid forecasted values with transform {1}'.format(len(invalid_idx), str(self.value_transformer)))
            for idx in invalid_idx:
                if idx == 0:
                    last_value = self.best_model.data.endog[-1]
                    last_value = self.value_transformer.inverse(last_value)
                    result[idx] = last_value
                else:
                    result[idx] = result[idx - 1]

        return result

    '''
    This method is specifically for generating y_output values in the horizon of ETS-CNN cnn model samples
    '''
    def get_horizon_seasonal_levels(self, horizon, start_index=None):
        start_index = len(self.etsfit_mtx) if start_index is None else start_index
        if start_index > len(self.etsfit_mtx):
            raise Exception('The starting point is beyond the end of time series!')

        seasonal_levels = np.repeat(self.etsfit_mtx[start_index-1, 1], horizon)
        season_component = self.get_fitted_season_component()
        if self.season_period is not None and season_component is not None:
            num_period = math.ceil(horizon/self.season_period)
            one_season = np.zeros(self.season_period)
            if start_index < self.season_period:
                one_season[-start_index:] = self.etsfit_mtx[0:start_index, 2]
            else:
                one_season = self.etsfit_mtx[start_index-self.season_period:start_index, 2]
            last_season_values = np.tile(one_season, num_period)[:horizon]

            if season_component == 'add':
                seasonal_levels += last_season_values
            elif season_component == 'mul':
                seasonal_levels *= last_season_values

        return seasonal_levels

    # update the self.etsfit_mtx using the best config
    def fit_delta(self, tsdelta):
        if self.best_config is None:
            raise Exception('the TS has not been fitted!')

        data = np.concatenate([self.proc_ts.values, tsdelta])
        t, s, sp, d, metric = self.best_config

        etsfit = self._fit_one_config(data, t, s, sp, d)
        if etsfit is None:
            # the old config does not work
            warnings.warn('old ets config cannot be applied to new delta data, using grid search instead!')
            # try grid search
            self.fit_grid_search()
        else:
            self.best_model = etsfit
            # update the best metric
            # there is no need to fill the ets_level_season_matrix, do it only when generating cnn samples
            self.best_metric = getattr(etsfit, self.criterion)

    def get_fitted_season_component(self):
        if self.best_config is None:
            raise Exception('The ets model has to be fitted first')

        return self.best_config[1]

    '''
    de/serialize method to reconstruct instance from file, need to load self.tsarray and self.best_config at least
    '''
    @classmethod
    def from_file(cls):
        pass

    # DEPRECATED this combination searches every possible combinations
    def get_config_all_comb(self):
        trend_params = ['add', 'mul', None]
        damped_params = [True, False]
        season_params = ['add', 'mul', None]
        season_period_params = {None}
        season_period_params.add(self.season_period)

        configs = list()
        for t in trend_params:
            for d in damped_params:
                for s in season_params:
                    for sp in season_period_params:
                        if s is None and sp is not None:
                            continue
                        # if s is not None and sp is not None:
                        configs.append([t, d, s, sp])

        return configs

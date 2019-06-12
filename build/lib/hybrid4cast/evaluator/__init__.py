import os
import abc
import numpy as np
import pandas as pd


class BACampaignEvaluator(object):

    def __init__(self, data_root, pp_datafile, use_transform='log'):
        self.data_root = data_root
        self.pp_datafile = pp_datafile

        self.season_period = 7
        self.sample_horizon = 30
        self.long_season_threshold = 30

        self.use_transform = use_transform

        self.hybrid_pipeline = None

    @property
    def absfp_pp_datafile(self):
        return os.path.join(self.data_root, self.pp_datafile)

    @abc.abstractmethod
    def absfp_ets_error(self):
        pass

    @abc.abstractmethod
    def absfp_etsfit(self):
        pass

    @abc.abstractmethod
    def absfp_hybrid_error(self):
        pass

    @abc.abstractmethod
    def load_data(self, data_type='Test'):
        pass

    @abc.abstractmethod
    def generate_ets_model(self, damped=True):
        pass

    @abc.abstractmethod
    def make_ets_forecast(self, etsfit_data):
        pass

    @abc.abstractmethod
    def run_hybrid_model(self, etsfit_data):
        pass


class M4Evaluator(object):

    m4_ts_types = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]

    def __init__(self, data_root, ets_output=None, hybrid_output=None, use_transform='log'):
        self.data_root = data_root
        ets_output = 'etsResult' if ets_output is None else ets_output
        hybrid_output = 'hybridResult' if hybrid_output is None else hybrid_output
        self.ets_output_folder = os.path.join(self.data_root, ets_output)
        self.hybrid_output_folder = os.path.join(self.data_root, hybrid_output)

        self.use_transform = use_transform

        self.m4_category = get_m4_category()
        self.hybrid_pipeline = None

    @abc.abstractmethod
    def absfp_ets_error(self, m4_type):
        pass

    @abc.abstractmethod
    def absfp_etsfit(self, m4_type):
        pass

    @abc.abstractmethod
    def absfp_hybrid_error(self, m4_type):
        pass

    @abc.abstractmethod
    def absfp_hybrid_tuning_history(self, m4_type):
        pass

    @abc.abstractmethod
    def load_data(self, m4_type='Hourly', data_type='Test'):
        pass

    @abc.abstractmethod
    def run_pure_ets(self, m4_type='Yearly', damped=True):
        '''
        :param m4_type: the m4 time series category
        :param damped: whether to explore damped es model
        :return:
        '''
        pass

    @abc.abstractmethod
    def run_hybrid_model(self, m4_type):
        pass


def get_m4_category():
    m4_types = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    num_ts_per_partition = 10
    num_minimum_partition = 100

    m4cat = pd.DataFrame(m4_types, index=[m4t[0] for m4t in m4_types], columns=['type'])
    m4cat['basea'] = [1, 4, 12, 4, 7, 24]
    m4cat['horizon'] = [6, 8, 18, 13, 14, 48]

    m4cat['window'] = [30, 40, 60, 60, 80, 168]
    m4cat['trastride'] = [6, 4, 6, 4, 4, 12]
    m4cat['longsea'] = [3, 3 * 4, 2 * 12, 12, 30, 24 * 7]

    ts_counts = [23000, 24000, 48000, 359, 4227, 414]
    ts_partitions = [np.max([np.ceil(ct / num_ts_per_partition), num_minimum_partition]) for ct in ts_counts]
    m4cat['count'] = ts_counts
    m4cat['spkpart'] = ts_partitions

    return m4cat

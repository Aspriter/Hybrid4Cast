import os
import time
import pickle
import numpy as np
import pandas as pd

from . import M4Evaluator

from ..hybrid.pipelines_spark import HybridModelPipelineSpark
from ..utils import print_section_msg, show_value_quantiles


class M4EvaluatorSpark(M4Evaluator):

    num_ts_per_partition = 10
    num_minimum_partition = 100

    def __init__(self, data_root, ets_output=None, hybrid_output=None, use_transform='log', spark_context=None):
        super(M4EvaluatorSpark, self).__init__(data_root, ets_output, hybrid_output, use_transform)

        if spark_context is None:
            raise Exception('The spark context cannot be None!')
        self.spark_context = spark_context

    def absfp_ets_error(self, m4_type):
        return os.path.join(self.ets_output_folder, m4_type + '-smape.pkl').replace('\\', '/')

    def absfp_etsfit(self, m4_type):
        return os.path.join(self.ets_output_folder, m4_type + '-etsfit.pkl').replace('\\', '/')

    def absfp_hybrid_error(self, m4_type):
        return os.path.join(self.hybrid_output_folder, m4_type + '-smape.pkl').replace('\\', '/')

    def absfp_hybrid_tuning_history(self, m4_type):
        pass

    def load_data(self, m4_type='Weekly', data_type='Test'):
        if data_type not in ['Train', 'Test']:
            raise Exception('')
        if m4_type not in ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]:
            raise Exception('')
        abs_file_path = os.path.join(self.data_root, data_type, m4_type + '-' + data_type.lower() + '.csv')

        def mapper_parse_m4_value_string(str_array):
            m4_id = str_array[0]
            m4_ts = []
            for value_str in str_array[1:]:
                try:
                    m4_ts.append(float(value_str))
                except ValueError:
                    break

            m4_ts = np.atleast_2d(m4_ts).T
            return m4_id, m4_ts, None

        rdd = self.spark_context.textFile(abs_file_path) \
            .map(lambda line: line.replace('"', '').split(',')) \
            .filter(lambda line: line[0] != 'V1') \
            .map(mapper_parse_m4_value_string)

        return rdd

    def run_pure_ets(self, m4_type='Yearly', damped=True, num_partition=None):
        '''
        This is a distributed spark implementation of evaluating one m4 category time series using pure ets model, with
        the following steps
        1. (distributed) fit ets models per time series
        2. (distributed) make ets forecast
        3. (distributed) compute error
        :param spark_context: the spark context used to generate rdd version of prediction data
        :param num_partition: number of partitions when building the input rdd
        :return: the smape error in rdd format where each item is (m4id, smape, scale) tuple
        '''

        num_partition = self.m4_category['spkpart'][m4_type[0]] if num_partition is None else num_partition
        sample_horizon = self.m4_category['horizon'][m4_type[0]]
        season_period = self.m4_category['basea'][m4_type[0]]
        long_season_threshold = self.m4_category['longsea'][m4_type[0]]

        self.hybrid_pipeline = HybridModelPipelineSpark(
            sample_horizon, None, season_period, self.use_transform, damped,
            long_season_threshold=long_season_threshold
        )

        print_section_msg('fit ets to time series')
        train_rdd = self.load_data(m4_type, 'Train')
        train_rdd = train_rdd.repartition(int(num_partition))
        etsfit_rdd = self.hybrid_pipeline.ets_train_model(train_rdd)

        abs_etsfit_file = self.absfp_etsfit(m4_type)
        try:
            etsfit_rdd.saveAsPickleFile(abs_etsfit_file)
            print('saving ets models to {0} successful!'.format(abs_etsfit_file))
        except Exception as e:
            print(str(e))

        print_section_msg('making ets forecast & computing smape distributed wth RDD')
        # no need to trigger the transform right away
        prediction_rdd = self.hybrid_pipeline.ets_make_forecast(etsfit_rdd)
        actual_rdd = self.load_data(m4_type, 'Test')
        error_rdd = HybridModelPipelineSpark.compute_smape(prediction_rdd, actual_rdd, 1, m4_type=m4_type)

        abs_smape_file = self.absfp_ets_error(m4_type)
        try:
            error_rdd.saveAsPickleFile(abs_smape_file)
            print('saving smape result to {0} successful!'.format(abs_smape_file))
        except Exception as e:
            print(str(e))

        return error_rdd

    def save_m4cat_etsfit_locally(self, m4_type):
        abs_file_path = self.ets_output_folder.replace('\\', '/')
        if abs_file_path[-1] == '/':
            abs_file_path = abs_file_path[:-1]
        abs_file_path += '_local/' + m4_type + '-etsfit.pkl'
        if abs_file_path[0] == '/':
            abs_file_path = '/' + abs_file_path
        abs_file_path = abs_file_path.replace(':', '')

        try:
            etsfit_list = self.spark_context.pickleFile(self.absfp_etsfit(m4_type)).collect()
            pickle.dump(etsfit_list, open(abs_file_path, 'wb'), protocol=4)
            print('pickle dumping {} ts category to {} successfully'.format(m4_type, abs_file_path))
        except Exception as e:
            print(str(e))

    def save_all_etsfit_locally(self):
        for m4_type in self.m4_ts_types:
            self.save_m4cat_etsfit_locally(m4_type)

    def run_hybrid_model(self, m4_type, num_partition=4, fit_ets=False, damped_ets=True,
                         sample_hislen=None, base_season_policy='force', long_season_policy=None,
                         holdout_length=0, train_stride=None, num_val=1, num_train=None, train_random=True,
                         outlier_colidx_x=None, outlier_colidx_y=None, threshold=5, method='neutral',
                         num_filters=32, num_layers=2, dropout_rate=(0.1, 0.3), loss='mae',
                         learning_rate=1e-3, optimizer='adam', batch_size=None, epochs=60, use_callback=False,
                         bootstrap_sampler='median', verbose=1):
        '''
        This is a mixed implementation of evaluating one m4 category time series using cnn model spark hybrid and
        local train/forecasting. Specifically, this method contains the following phases:
        1. (distributed) fit ets models per time series
        2. (distributed/local) extract training samples, input_x and ets_normalizers for forecast and actual values
        3. (local) NN normalization, train model and make forecast
        4. (distributed) compute error
        :param spark_context: the spark context used to generate rdd version of prediction data
        :param m4_type: the m4 time series category
        :param num_partition: number of partitions when building the input rdd
        :param fit_ets: boolean to indicate whether to actually do ets fitting or load fitted result from dbfs
        :return: the smape error in rdd format where each item is (m4id, smape, scale) tuple
        '''

        sample_hislen = int(self.m4_category['window'][m4_type[0]]) if sample_hislen is None else sample_hislen
        sample_horizon = self.m4_category['horizon'][m4_type[0]]
        season_period = self.m4_category['basea'][m4_type[0]]
        train_stride = self.m4_category['trastride'][m4_type[0]] if train_stride is None else train_stride
        long_season_threshold = self.m4_category['longsea'][m4_type[0]]

        hybrid_pipeline = HybridModelPipelineSpark(
            sample_horizon, sample_hislen, season_period, self.use_transform, damped_ets,
            base_season_policy, long_season_threshold, long_season_policy
        )
        self.hybrid_pipeline = hybrid_pipeline

        abs_etsfit_file = self.absfp_etsfit(m4_type)
        if fit_ets:
            # fit ets models first per time series and save it
            print_section_msg('fit ets to time series')
            train_rdd = self.load_data(m4_type, 'Train')
            train_rdd = train_rdd.repartition(int(num_partition))
            etsfit_rdd = self.hybrid_pipeline.ets_train_model(train_rdd)

            etsfit_rdd.saveAsPickleFile(abs_etsfit_file)
        else:
            print_section_msg('load fitted ets models')
            etsfit_rdd = self.spark_context.pickleFile(abs_etsfit_file)

        # have to get input used for evaluation before model training, because keras model cannot be pickle serialized.
        # When calling rdd map, this instance will be serialized and passed to workers
        print_section_msg('prepare testing samples for evaluation')
        id_set, x_forecast, y_ets_normalizer, y_value_transformer = hybrid_pipeline.get_forecast_input(etsfit_rdd)

        # train the model
        print_section_msg('start the model training process')
        hybrid_pipeline.hybrid_train_model(
            etsfit_rdd, holdout_length, train_stride, num_val, num_train, train_random,
            outlier_colidx_x, outlier_colidx_y, threshold, method,
            num_filters, num_layers, dropout_rate, loss,
            learning_rate, optimizer, batch_size, epochs, use_callback,
            verbose_control=1
        )
        print('model training process info:')
        print(pd.DataFrame(hybrid_pipeline.cnn_model.history.history))

        # make predictions locally
        print_section_msg('start making local forecast')
        prediction_rdd = hybrid_pipeline.hybrid_make_forecast(x_forecast, y_ets_normalizer, y_value_transformer,
                                                              id_set=id_set, sc=self.spark_context)

        print_section_msg('computing smape of cnn model')
        actual_rdd = self.load_data(m4_type, 'Test')
        error_rdd = hybrid_pipeline.compute_smape(prediction_rdd, actual_rdd, 1, m4_type=m4_type)
        try:
            error_rdd.saveAsPickleFile(self.absfp_hybrid_error(m4_type))
        except Exception as e:
            print(str(e))

        return error_rdd

    def evaluate_all_m4(self, model='ets', **kwargs):
        if model not in ['ets', 'cnn']:
            raise ValueError('unrecognized model!')

        total_rdd = self.spark_context.emptyRDD()
        for one_m4 in self.m4_ts_types:
            t0 = time.time()
            num_partitions = self.m4_category['spkpart'][one_m4[0]]
            print_section_msg('evaluating {0} model on {1} ts'.format(model, one_m4))
            if model == 'ets':
                one_m4_error_rdd = self.run_pure_ets(one_m4, num_partition=num_partitions, **kwargs)
            else:
                one_m4_error_rdd = self.run_hybrid_model(one_m4, num_partition=num_partitions, **kwargs)

            total_rdd = total_rdd.union(one_m4_error_rdd)
            print('time it took {0}'.format(time.time() - t0))

        HybridModelPipelineSpark.show_smape_rdd(total_rdd, 'overall')

        return total_rdd

    def show_all_m4_smape(self, result_folder, m4_types=None):
        m4_types = self.m4_ts_types if m4_types is None else m4_types
        total_smape = []
        for one_m4 in m4_types:
            abs_file_path = os.path.join(self.data_root, result_folder, one_m4 + '-smape.pkl')
            try:
                smape_rdd = self.spark_context.pickleFile(abs_file_path)
                total_smape.append(HybridModelPipelineSpark.show_smape_rdd(smape_rdd, one_m4))
            except Exception as e:
                print(str(e))
                continue

        if len(total_smape) > 0:
            total_smape = np.concatenate(total_smape)
            print('overall {0} ts with mean smape: {1}'.format(len(total_smape), np.nanmean(total_smape)))
            show_value_quantiles(total_smape)

    def get_etsfit_details(self, m4_type):

        def mapper_extract_etsfit_detail(etsfit_result):
            m4_id, etsfits, ts_info = etsfit_result
            details = list()
            details.append(m4_id)
            details.append(etsfits[0].proc_ts.length)
            details.append(etsfits[0].season_period)
            details.append(etsfits[0].proc_ts.sig_long_season)
            details.append(etsfits[0].proc_ts.trend)
            for ets_component in etsfits[0].best_config:
                details.append(ets_component)
            return details

        etsfit_rdd = self.spark_context.pickleFile(self.absfp_etsfit(m4_type))
        etsfit_rdd = etsfit_rdd.map(mapper_extract_etsfit_detail)

        return pd.DataFrame(etsfit_rdd.collect(), \
                            columns=['m4_id', 'tslen', 'basea', 'longsea', 'trend', 't', 's', 'sp', 'd', 'ic'])



'''
draft for validation with Uber M4 data set
input: list of TS data matrix

1. train, val, test = []
for each TS:
    partition data into training/testing set
    generate train_per_cpg, val_per_cpg, test_x_input using training set, 
    save test set as real outpt for validation using testing set
    train.append(train_per_cpg)
    val.append(val_per_cpg)
    test.append(test_per_cpg)

2. use train, val to fit NN model, doing evaluation with test
'''
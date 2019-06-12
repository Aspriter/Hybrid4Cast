import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..hybrid.pipelines_spark import HybridModelPipelineSpark
from ..hybrid.pipelines_local import HybridModelPipelineLocal
from ..optimizer.hyper_para_optimizer import CNNOptimizer
from ..utils import print_section_msg


class BACampaignEvaluatorSpark:

    def __init__(self, data_root='d:/toguan/bac/', data_file='cpgperf_prod1209.tsv', use_transform='log'):
        self.data_root = data_root
        self.data_file = data_file

        self.season_period = 7
        self.sample_horizon = 30
        self.long_season_threshold = 30
        self.use_transform = use_transform

        self.hybrid_pipeline = None

    @property
    def absfp_data_file(self):
        return os.path.join(self.data_root, self.data_file)

    @property
    def absfp_ets_error(self):
        return os.path.join(self.data_root, 'ets-smape.pkl').replace('\\', '/')

    @property
    def absfp_ets_error_local(self):
        return os.path.join(self.data_root, 'ets-smape.csv')

    @property
    def absfp_hybrid_error(self):
        return os.path.join(self.data_root, 'cnn-smape.csv')

    @property
    def absfp_etsfit(self):
        return os.path.join(self.data_root, 'ets-fit.pkl').replace('\\', '/')

    def save_etsfit_locally(self, spark_context):
        abs_file_path = '/' + self.data_root + 'ets-fit-local.pkl'
        abs_file_path = abs_file_path.replace(':', '')

        try:
            etsfit_list = spark_context.pickleFile(self.absfp_etsfit).collect()
            pickle.dump(etsfit_list, open(abs_file_path, 'wb'), protocol=4)
            print('pickle dumping to {} successfully'.format(abs_file_path))
        except Exception as e:
            print(str(e))

    def run_pure_ets_evaluation(self, spark_context, num_partition=1000, holdout_length=30, damped_ets=False):

        self.hybrid_pipeline = HybridModelPipelineSpark(
            self.sample_horizon, None, self.season_period, self.use_transform, damped_ets,
            long_season_threshold=self.long_season_threshold
        )

        print_section_msg('fit ets to time series')
        train_rdd = self.load_data(spark_context, 'Train', holdout_length)
        train_rdd = train_rdd.repartition(int(num_partition))
        etsfit_rdd = self.hybrid_pipeline.ets_train_model(train_rdd)

        try:
            etsfit_rdd.saveAsPickleFile(self.absfp_etsfit)
            print('saving ets models to {0} successful!'.format(self.absfp_etsfit))
        except:
            pass

        # print_section_msg('making ets forecast & computing smape distributed wth RDD')
        # # no need to trigger the transform right away
        # prediction_rdd = self.hybrid_pipeline.ets_make_forecast(etsfit_rdd)
        # actual_rdd = self.load_data(spark_context, 'Test', holdout_length)
        # error_rdd = HybridModelPipelineSpark.compute_smape(prediction_rdd, actual_rdd, 1)
        #
        # try:
        #     error_rdd.saveAsPickleFile(self.absfp_ets_error)
        #     print('saving smape result to {0} successful!'.format(self.absfp_ets_error))
        # except:
        #     pass

        return etsfit_rdd

    def evaluate_local_ets(self, etsfit_list, holdout_length=60):
        warnings.filterwarnings('ignore')
        t0 = time.time()

        hybrid_pipeline = HybridModelPipelineLocal(
            self.sample_horizon, None, self.season_period, self.use_transform, False,
            long_season_threshold=self.long_season_threshold
        )
        self.hybrid_pipeline = hybrid_pipeline

        print_section_msg('make ets forecast and do evaluation')
        id_set, ets_prediction = hybrid_pipeline.ets_make_forecast(etsfit_list)

        print('loading test data...')
        test_data_dict = pickle.load(open(os.path.join(self.data_root, 'test_data_{}.pkl'.format(holdout_length)), 'rb'))
        test_actual = np.array([test_data_dict[uid] for uid in id_set])
        smape_table = hybrid_pipeline.compute_smape(ets_prediction, test_actual, 1)

        print('save ets err table to {0}'.format(self.absfp_ets_error_local))
        smape_table.to_csv(self.absfp_ets_error_local, index=False)

        print('Time it took {0}'.format(time.time() - t0))
        warnings.filterwarnings('default')
        return smape_table

    @staticmethod
    def mapper_parse_row_string(row_str):
        colidx = [0, 9, 11]
        fields = row_str.split('\t')
        uid = int(fields[0])
        ts_mtx = np.array([np.array(odstr.split(","))[colidx] for odstr in fields[1].split(";")])
        ts_mtx = ts_mtx[np.argsort(ts_mtx[:, 0])]
        ts_mtx = ts_mtx[:, [1, 2]]
        ts_mtx = ts_mtx.astype(np.float)
        return uid, ts_mtx

    def load_data(self, spark_context, data_type='Test', holdout_length=None):
        if data_type not in ['Train', 'Test']:
            raise Exception('')

        holdout_length = self.sample_horizon if holdout_length is None else holdout_length

        def mapper_parse_row_string(row_str):
            uid, ts_mtx = BACampaignEvaluatorSpark.mapper_parse_row_string(row_str)

            if data_type == 'Train':
                ts_mtx = ts_mtx[:-holdout_length]
            else:
                if holdout_length != self.sample_horizon:
                    ts_mtx = ts_mtx[-holdout_length:self.sample_horizon-holdout_length]
                else:
                    ts_mtx = ts_mtx[-holdout_length:]

            return uid, ts_mtx

        rdd = spark_context.textFile(self.absfp_data_file).map(mapper_parse_row_string)

        return rdd

    def load_local_test_data(self, holdout_length=None):
        holdout_length = self.sample_horizon if holdout_length is None else holdout_length

        test_dict = dict()
        datafile = open(self.absfp_data_file, 'r')
        progress_bar = tqdm(datafile, desc='processing', unit=' cpg', leave=True, ascii=True, ncols=105)
        for line in datafile:
            progress_bar.update(1)

            uid, ts_mtx = BACampaignEvaluatorSpark.mapper_parse_row_string(line)
            if holdout_length != self.sample_horizon:
                ts_mtx = ts_mtx[-holdout_length:self.sample_horizon-holdout_length]
            else:
                ts_mtx = ts_mtx[-holdout_length:]
            # pdb.set_trace()
            test_dict[uid] = ts_mtx

        progress_bar.close()
        datafile.close()

        absfp_test_data = os.path.join(self.data_root, 'test_data_{}.pkl'.format(holdout_length))
        pickle.dump(test_dict, open(absfp_test_data, 'wb'), protocol=4)

    def tune_hyper_parameters(self, etsfit_file, n_calls=20, **kwargs):
        t0 = time.time()

        absfp_etsfit = os.path.join(self.data_root, etsfit_file)
        print_section_msg('loading fitted ets model file {}'.format(absfp_etsfit))
        with open(absfp_etsfit, 'rb') as fid:
            etsfit_list = pickle.load(fid)

        hybrid_pipeline = HybridModelPipelineLocal(self.sample_horizon, None, self.season_period, 'log', True,
                                                   'byets', self.long_season_threshold, None)

        actual_value_id = np.array([etsfit[0] for etsfit in etsfit_list])
        test_data_dict = pickle.load(open(os.path.join(self.data_root, 'test_data_60.pkl'), 'rb'))
        y_actual = np.array([test_data_dict[uid] for uid in actual_value_id])

        # actual_value_id, y_actual, y_ets_scaled = hybrid_pipeline.get_actual_output(etsfit_list, self.sample_horizon)
        # y_actual = np.expm1(y_actual)
        print('validation data set shape {0}'.format(y_actual.shape))

        hyper_para_space = CNNOptimizer.get_bac_hyper_space()
        res_gp = CNNOptimizer.optimize(
            hyper_para_space, hybrid_pipeline, etsfit_list, (actual_value_id, y_actual), n_calls,
            y_val=None, train_stride=7, holdout_length=0, obj_ratio=1, **kwargs
        )

        output_file = os.path.join(self.data_root, 'tune-result.pkl')
        if 'all_random' in kwargs.keys() and kwargs['all_random']:
            output_file = output_file[:-4] + '-init.pkl'
        pickle.dump(res_gp, open(output_file, 'wb'), protocol=4)

        print('Time it took {0}'.format(time.time() - t0))

        return res_gp

    def run_hybrid_model(self, etsfit_file, sample_hislen=90, base_season_policy='byets', long_season_policy=None,
                         holdout_length=0, train_stride=7, num_val=0, num_train=None, train_random=True,
                         outlier_colidx_x=[0, 1], outlier_colidx_y=[0, 1], threshold=50, method='neutral',
                         num_filters=32, num_layers=2, dropout_rate=(0.1, 0.3), loss='mae',
                         learning_rate=1e-3, optimizer='adam', batch_size=256, epochs=30, use_callback=False,
                         bootstrap_sampler='median', verbose=1, error_file=True):
        t0 = time.time()

        absfp_etsfit = os.path.join(self.data_root, etsfit_file)
        print_section_msg('loading fitted ets model file {}'.format(absfp_etsfit))
        with open(absfp_etsfit, 'rb') as fid:
            etsfit_list = pickle.load(fid)

        hybrid_pipeline = HybridModelPipelineLocal(
            self.sample_horizon, sample_hislen, self.season_period, self.use_transform, True,
            base_season_policy, self.long_season_threshold, long_season_policy
        )
        self.hybrid_pipeline = hybrid_pipeline

        print_section_msg('excute training process')
        if long_season_policy is not None:
            outlier_colidx_x = list(range(len(outlier_colidx_y)))
        hybrid_pipeline.hybrid_train_model(
            etsfit_list, holdout_length, train_stride, num_val, num_train, train_random,
            outlier_colidx_x, outlier_colidx_y, threshold, method,
            num_filters, num_layers, dropout_rate, loss,
            learning_rate, optimizer, batch_size, epochs, use_callback,
            verbose_control=verbose, verbose_nn=0
        )

        if verbose:
            train_process_table = pd.DataFrame(hybrid_pipeline.cnn_model.history.history)
            print('model training process info:')
            print(train_process_table)

        print_section_msg('generating test samples')
        id_set, x_forecast_set, y_ets_normalizer_set, value_transformer_set = \
            hybrid_pipeline.get_forecast_input(etsfit_list, 0, verbose)

        print_section_msg('making forecast')
        y_prediction = hybrid_pipeline.hybrid_make_forecast(x_forecast_set, y_ets_normalizer_set, value_transformer_set,
                                                            bootstrap_sampler=bootstrap_sampler, verbose=verbose)

        print_section_msg('computing smape')
        test_data_dict = pickle.load(open(os.path.join(self.data_root, 'test_data_60.pkl'), 'rb'))
        test_actual = np.array([test_data_dict[uid] for uid in id_set])
        smape_table = hybrid_pipeline.compute_smape(y_prediction, test_actual, 1)

        if isinstance(error_file, bool) and not error_file:
            pass
        else:
            print_section_msg('storing results')
            if not isinstance(error_file, str):
                error_file = self.absfp_hybrid_error

            print('save cnn smape err table to {0}'.format(error_file))
            smape_table.to_csv(error_file, index=False)

        print('Time it took {0}'.format(time.time() - t0))

        return smape_table

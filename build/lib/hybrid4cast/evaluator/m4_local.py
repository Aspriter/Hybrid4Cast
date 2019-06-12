import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from inspect import signature
from tqdm import tqdm

from . import M4Evaluator

from ..classicts.ets_forecaster import ETSForecaster
from ..hybrid.pipelines_local import HybridModelPipelineLocal
from ..utils import print_section_msg, do_para_call
from ..optimizer.hyper_para_optimizer import CNNOptimizer


class M4EvaluatorLocal(M4Evaluator):

    best_model_temp_file = 'best_model_temp_file'

    def __init__(self, data_root='d:/toguan/UberM4/', ets_output=None, hybrid_output=None, use_transform='log'):
        super(M4EvaluatorLocal, self).__init__(data_root, ets_output, hybrid_output, use_transform)

        if not os.path.exists(self.ets_output_folder):
            os.makedirs(self.ets_output_folder)
        if not os.path.exists(self.hybrid_output_folder):
            os.makedirs(self.hybrid_output_folder)

    def absfp_ets_error(self, m4_type):
        return os.path.join(self.ets_output_folder, m4_type + '-smape.csv')

    def absfp_etsfit(self, m4_type):
        return os.path.join(self.ets_output_folder, m4_type + '-etsfit.pkl')

    def absfp_hybrid_error(self, m4_type):
        return os.path.join(self.hybrid_output_folder, m4_type + '-smape.csv')

    def absfp_hybrid_tuning_history(self, m4_type):
        return os.path.join(self.hybrid_output_folder, m4_type + '-tune.pkl')

    def load_data(self, m4_type='Hourly', data_type='Test'):
        if data_type not in ['Train', 'Test']:
            raise Exception('')
        if m4_type not in self.m4_ts_types:
            raise Exception('')
        abs_file_path = os.path.join(self.data_root, data_type, m4_type + '-' + data_type.lower() + '.csv')

        data_mtx = pd.read_csv(abs_file_path)
        data_mtx = data_mtx.set_index('V1')
        return data_mtx

    def run_pure_ets(self, m4_type='Hourly', damped=True, fit_model=True):
        warnings.filterwarnings('ignore')
        t0 = time.time()
        print_section_msg('evaluating pure ETS on Uber M4 {0} time series'.format(m4_type))

        horizon = self.m4_category['horizon'][m4_type[0]]
        season_period = self.m4_category['basea'][m4_type[0]]
        long_season_threshold = self.m4_category['longsea'][m4_type[0]]

        hybrid_pipeline = HybridModelPipelineLocal(
            horizon, None, season_period, self.use_transform, damped,
            long_season_threshold=long_season_threshold
        )
        self.hybrid_pipeline = hybrid_pipeline

        if fit_model:
            print('loading training data...')
            train_data_all = self.load_data(m4_type, 'Train')

            input_data = list()
            for m4_id, train_data in train_data_all.iterrows():
                train_data = train_data[np.isfinite(train_data)]
                train_data = np.atleast_2d(train_data).T
                input_data.append((m4_id, train_data))

            print_section_msg('start fitting ets models')
            etsfit_list = hybrid_pipeline.ets_train_model(input_data, verbose=1)
            print('save ets fitted model to {0}'.format(self.absfp_etsfit(m4_type)))
            with open(self.absfp_etsfit(m4_type), 'wb') as fid:
                pickle.dump(etsfit_list, fid, protocol=4)
        else:
            print_section_msg('loading fitted ets models from pickle file')
            with open(self.absfp_etsfit(m4_type), 'rb') as fid:
                etsfit_list = pickle.load(fid)

        print_section_msg('make ets forecast and do evaluation')
        id_set, ets_prediction = hybrid_pipeline.ets_make_forecast(etsfit_list)

        print('loading test data...')
        test_data_all = self.load_data(m4_type, 'Test')
        id_actual = test_data_all.index.values
        test_actual = np.expand_dims(test_data_all.values, axis=2)
        smape_table = hybrid_pipeline.compute_smape(ets_prediction, test_actual, verbose=1,
                                                    id_prediction=id_set, id_actual=id_actual)

        print('save ets err table to {0}'.format(self.absfp_ets_error(m4_type)))
        smape_table.to_csv(self.absfp_ets_error(m4_type), index=False)

        print('Time it took {0}'.format(time.time() - t0))
        warnings.filterwarnings('default')
        return smape_table

    def load_etsfit_r_output(self, data_file, m4_type='Hourly'):
        t0 = time.time()
        print('loading {0} etsfit models from R cosmos output tsv'.format(m4_type))

        m4_cat_long_season_threshold = self.m4_category['longsea'][m4_type[0]]
        m4_cat_horizon = int(self.m4_category['horizon'][m4_type[0]])

        etsfit_list = list()
        progress_bar = tqdm(total=self.m4_category['count'][m4_type[0]], desc='processing Uber M4 times series',
                            unit='ts', leave=True, ascii=True, ncols=105)
        for line in open(data_file, 'r'):
            if line[0] != m4_type[0]:
                continue

            progress_bar.update(1)
            cells = line.split('\t')

            m4_id = cells[0]
            etsfit = ETSForecaster.from_r_result_tsv(cells[3], cells[2])

            etsfit.proc_ts.long_season_threshold = m4_cat_long_season_threshold
            etsfit.etsfit_mtx = etsfit.etsfit_mtx[:-m4_cat_horizon]
            etsfit.proc_ts.values = etsfit.proc_ts.values[:-m4_cat_horizon]
            etsfit.proc_ts.length -= m4_cat_horizon

            etsfit_list.append((m4_id, [etsfit]))
        progress_bar.close()

        print('save ets fitted model to {0}'.format(self.absfp_etsfit(m4_type)))
        with open(self.absfp_etsfit(m4_type), 'wb') as fid:
            pickle.dump(etsfit_list, fid, protocol=4)

        print('Time it took {0}'.format(time.time() - t0))

    def run_hybrid_model(self, m4_type='Hourly', sample_hislen=None, base_season_policy='byets', long_season_policy=None,
                         holdout_length=0, train_stride=None, num_val=0, num_train=None, train_random=True,
                         outlier_colidx_x=[0], outlier_colidx_y=[0], threshold=None, method='neutral',
                         num_filters=32, num_layers=2, dropout_rate=(0.1, 0.3), loss='mae',
                         learning_rate=1e-3, optimizer='adam', batch_size=256, epochs=60, use_callback=False,
                         bootstrap_sampler='median', verbose=1, error_file=True):
        t0 = time.time()

        # load fitted ets models from disk
        print_section_msg('loading fitted ets models from pickle file')
        with open(self.absfp_etsfit(m4_type), 'rb') as fid:
            etsfit_list = pickle.load(fid)

        sample_hislen = self.m4_category['window'][m4_type[0]] if sample_hislen is None else sample_hislen
        sample_horizon = self.m4_category['horizon'][m4_type[0]]
        season_period = self.m4_category['basea'][m4_type[0]]
        train_stride = self.m4_category['trastride'][m4_type[0]] if train_stride is None else train_stride
        long_season_threshold = self.m4_category['longsea'][m4_type[0]]

        hybrid_pipeline = HybridModelPipelineLocal(sample_horizon, sample_hislen, season_period, self.use_transform, True,
                                                   base_season_policy, long_season_threshold, long_season_policy)
        self.hybrid_pipeline = hybrid_pipeline

        print_section_msg('excute training process')
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
        test_actual = self.load_data(m4_type, 'Test')
        id_actual = test_actual.index.values
        test_actual = np.expand_dims(test_actual.values, axis=2)
        smape_table = hybrid_pipeline.compute_smape(y_prediction, test_actual, 1,
                                                    id_prediction=id_set, id_actual=id_actual)

        if isinstance(error_file, bool) and not error_file:
            pass
        else:
            print_section_msg('storing results')
            if not isinstance(error_file, str):
                error_file = self.absfp_hybrid_error(m4_type)

            print('save cnn smape err table to {0}'.format(error_file))
            smape_table.to_csv(error_file, index=False)

        print('Time it took {0}'.format(time.time() - t0))

        return smape_table

    def run_batch_optimal_hybrid(self, m4_type, iteration=10, **kwargs):
        res_tune = pickle.load(open(self.absfp_hybrid_tuning_history(m4_type), 'rb'))

        output_folder = os.path.join(self.hybrid_output_folder, m4_type)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        smape_list = []
        for i in range(iteration):
            error_file = os.path.join(output_folder, 'smape{}.csv'.format(i))
            one_smape = self.run_optimal_hybrid(m4_type, res_gp=res_tune, verbose=0, error_file=error_file, **kwargs)
            smape_list.append(np.mean(one_smape.smape.values))
            print('\n')

        print('{} iterations mean smape {}, sd {}'.format(iteration, np.mean(smape_list), np.std(smape_list)))

        return smape_list

    def run_optimal_hybrid(self, m4_type, **kwargs):
        if 'res_gp' in kwargs.keys():
            res_gp = kwargs['res_gp']
            kwargs.pop('res_gp')
        else:
            res_gp = pickle.load(open(self.absfp_hybrid_tuning_history(m4_type), 'rb'))

        para_dict = CNNOptimizer.get_para_dict(res_gp, **kwargs)

        hybrid_func_para = signature(self.run_hybrid_model).parameters.keys()
        for para_key in kwargs.keys():
            if para_key in hybrid_func_para:
                para_dict[para_key] = kwargs[para_key]
        print('do call cnn model for {} with optimal para set:\n {}\n'.format(m4_type, para_dict))

        para_dict['m4_type'] = m4_type
        return do_para_call(self.run_hybrid_model, **para_dict)

    def tune_hyper_parameters(self, m4_type='Yearly', n_calls=50, **kwargs):
        t0 = time.time()

        print_section_msg('loading fitted ets models of {0} ts from pickle file'.format(m4_type))
        with open(self.absfp_etsfit(m4_type), 'rb') as fid:
            etsfit_list = pickle.load(fid)

        m4_optimizer = CNNOptimizer(etsfit_list)

        hyper_para_space = CNNOptimizer.get_m4_hyper_space(m4_type)

        sample_horizon = self.m4_category['horizon'][m4_type[0]]
        season_period = self.m4_category['basea'][m4_type[0]]
        long_season_threshold = self.m4_category['longsea'][m4_type[0]]
        train_stride = self.m4_category['trastride'][m4_type[0]]

        hybrid_pipeline = HybridModelPipelineLocal(sample_horizon, None, season_period, 'log', True,
                                                   'byets', long_season_threshold, None)

        actual_value_id, y_actual, y_ets_scaled = hybrid_pipeline.get_actual_output(etsfit_list, sample_horizon, verbose=1)
        y_actual = np.expm1(y_actual)
        print('validation data set shape {0}'.format(y_ets_scaled.shape))

        res_gp = CNNOptimizer.optimize(
            hyper_para_space, hybrid_pipeline, etsfit_list, (actual_value_id, y_actual),
            n_calls, y_val=y_ets_scaled, train_stride=train_stride, holdout_length=sample_horizon, **kwargs
        )

        output_file = self.absfp_hybrid_tuning_history(m4_type)
        if 'all_random' in kwargs.keys() and kwargs['all_random']:
            output_file = output_file[:-4] + '-init.pkl'
        pickle.dump(res_gp, open(output_file, 'wb'), protocol=4)

        print('Time it took {0}'.format(time.time() - t0))

        return res_gp

    def tune_all(self):

        num_init = 30

        # tune_y = self.tune_hyper_parameters('Yearly', num_init, all_random=True)
        # tune_q = self.tune_hyper_parameters('Quarterly', num_init, all_random=True)
        # tune_m = self.tune_hyper_parameters('Monthly', num_init, all_random=True)

        tune_d = self.tune_hyper_parameters('Daily', num_init, all_random=True)
        tune_w = self.tune_hyper_parameters('Weekly', num_init, all_random=True)
        tune_h = self.tune_hyper_parameters('Hourly', num_init, all_random=True)



import time
import numpy as np
import pandas as pd

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Space, Integer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from skopt.utils import normalize_dimensions

from ..hybrid.pipelines_local import HybridModelPipelineLocal
from ..cnn.hybrid_model_forecaster import HybridModelForecaster
from ..utils import print_section_msg, do_para_call


class CNNOptimizer:

    num_early_stop = 4
    norm_threshold_values = [20, 50, 100]

    def __init__(self, etsfit_list=None):
        self.m4_bound = self.get_m4_cat_hyper_para_bounds()

        self.etsfit_list = etsfit_list

    @staticmethod
    def get_m4_cat_hyper_para_bounds():
        m4_types = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
        m4bound = pd.DataFrame(m4_types, index=[m4t[0] for m4t in m4_types], columns=['type'])
        m4bound['stride_l'] = [2, 2, 3, 4, 7, 6]
        m4bound['stride_h'] = [6, 8, 24, 16, 14, 12]

        m4bound['hislen_l'] = [18, 12, 36, 12, 21, 72]
        m4bound['hislen_h'] = [30, 20, 60, 60, 35, 120]

        m4bound['nlayer_h'] = [4, 6, 8, 4, 4, 4]

        return m4bound

    @staticmethod
    def get_para_dict(res_gp, **kwargs):
        para_dict = dict()
        if 'para_space' in kwargs.keys():
            para_space = kwargs['para_space']
        else:
            para_space = res_gp.space.dimensions

        sel_para_index = np.argmin(res_gp.func_vals)
        if 'iter_index' in kwargs.keys():
            sel_para_index = kwargs['iter_index']

        para_values = res_gp.x_iters[sel_para_index]
        for one_para, value in zip(para_space, para_values):
            if one_para.name == 'threshold':
                para_dict[one_para.name] = CNNOptimizer.norm_threshold_values[value]
            else:
                para_dict[one_para.name] = value

        if 'epochs' not in para_dict.keys() and 'epochs' in res_gp.specs.keys():
            if 'val_loss' in res_gp.specs.keys():
                sel_para_index -= len(res_gp.func_vals) - len(res_gp.specs['val_loss'])
            para_dict['epochs'] = res_gp.specs['epochs'][sel_para_index]

        return para_dict

    @staticmethod
    def show_para_search_history(res_gp):
        col_names = [one_space.name for one_space in res_gp.space.dimensions]
        his_table = pd.DataFrame(res_gp.x_iters, columns=col_names)
        if 'threshold' in his_table.columns:
            his_table['threshold'] = [CNNOptimizer.norm_threshold_values[vi] for vi in his_table.threshold]
        his_table['func_val'] = res_gp.func_vals
        his_table = his_table.sort_values('func_val')
        return his_table

    @staticmethod
    def get_m4_hyper_space(m4_type):
        m4t = m4_type[0]

        m4bd = CNNOptimizer.m4_bound
        hyper_para_space = list()
        # hyper_para_space.append(Categorical(['mae', 'mse', 'gaussian'], name='loss'))
        hyper_para_space.append(Integer(10, 30, name='epochs'))
        # hyper_para_space.append(Real(1e-4, 1e-2, name='learning_rate'))
        # hyper_para_space.append(Categorical(['trim', 'neutral'], name='method'))
        hyper_para_space.append(Integer(0, 2, name='threshold'))

        num_layer_up = np.ceil(np.log2(m4bd['hislen_h'][m4t]))
        num_layer_up = np.min([num_layer_up, m4bd['nlayer_h'][m4t]])
        hyper_para_space.append(Integer(2, num_layer_up, name='num_layers'))
        # hyper_para_space.append(Integer(32, 64, name='num_filters'))
        # hyper_para_space.append(Integer(32, 256, name='batch_size'))
        # hyper_para_space.append(Real(0.2, 0.5, name='dropout_rate'))
        if m4_type not in ['Yearly', 'Weekly', 'Hourly']:
            hyper_para_space.append(Integer(3, 11, name='num_train'))
        # hyper_para_space.append(Integer(m4bd['stride_l'][m4t], m4bd['stride_h'][m4t], name='train_stride'))
        hyper_para_space.append(Integer(m4bd['hislen_l'][m4t], m4bd['hislen_h'][m4t], name='sample_hislen'))

        return hyper_para_space

    @staticmethod
    def get_bac_hyper_space():
        hyper_para_space = list()
        hyper_para_space.append(Integer(10, 30, name='epochs'))
        hyper_para_space.append(Integer(0, 2, name='threshold'))

        hyper_para_space.append(Integer(2, 6, name='num_layers'))
        hyper_para_space.append(Integer(5, 13, name='num_train'))
        hyper_para_space.append(Integer(21, 105, name='sample_hislen'))

        return hyper_para_space

    # def optimize(self, m4_type, n_calls=30, **kwargs):
    @staticmethod
    def optimize(para_space, hybrid_pipeline, etsfit_list, y_actual_suit, n_calls=10, y_val=None, **kwargs):

        actual_value_id, y_actual = y_actual_suit
        num_target = y_actual.shape[-1]
        x_target_colidx = list(range(num_target))
        # x_target_colidx = list(range(num_target * 2))
        y_target_colidx = list(range(num_target))
        train_stride, holdout_length, obj_ratio = None, 0, 1
        if 'train_stride' in kwargs.keys():
            train_stride = kwargs['train_stride']
        if 'holdout_length' in kwargs.keys():
            holdout_length = kwargs['holdout_length']
        if 'obj_ratio' in kwargs.keys():
            obj_ratio = kwargs['obj_ratio']
        epochs_list, trapara_ratio_list, val_loss_list = [], [], []

        @use_named_args(para_space)
        def objective(**params):
            t0 = time.time()
            print_section_msg('new iteration seperator')
            print('hyper parameters: {0}'.format(params))

            hybrid_pipeline.set_hybrid_sample_history_length(params['sample_hislen'])

            this_train_stride = train_stride
            if 'train_stride' in params.keys():
                this_train_stride = params['train_stride']
                params.pop('train_stride')
            x_train_set, y_train_set, _, _ = do_para_call(
                hybrid_pipeline.generate_training_samples, etsfit_list,
                holdout_length=holdout_length, num_val=0, train_stride=this_train_stride, verbose=0, **params
            )

            input_shape = x_train_set.shape[1:]
            output_shape = y_train_set.shape[1:]
            cnn_model = HybridModelForecaster()
            do_para_call(cnn_model.build_model, input_shape, output_shape, **params)
            cnn_param_ct = cnn_model.get_param_count()
            train_param_ratio = float(len(x_train_set)) / cnn_param_ct
            trapara_ratio_list.append(train_param_ratio)
            print('{} cnn params while {} training samples, ratio {}'.format(cnn_param_ct, len(x_train_set), train_param_ratio))
            # if train_param_ratio < 10:
            #     val_loss_list.append([])
            #     epochs_list.append(0)
            #     return 1e3

            id_set, x_forecast_set, y_ets_normalizer_set, value_transformer_set = hybrid_pipeline.get_forecast_input(
                etsfit_list, holdout_length, verbose=0
            )

            # do normalization
            threshold_value = None
            if 'threshold' in params.keys():
                threshold_value = CNNOptimizer.norm_threshold_values[params['threshold']]
                params.pop('threshold')
            x_train_set, y_train_set, _, _ = do_para_call(
                hybrid_pipeline.normalize_nn, x_train_set, y_train_set,
                outlier_colidx_x=x_target_colidx, outlier_colidx_y=y_target_colidx, threshold=threshold_value, **params
            )

            this_x_val, this_y_val = None, None
            if y_val is not None:
                # asymmetric normalization,
                normalizer = hybrid_pipeline.normalizer
                # for input x, using defined outlier_colidx/threshold/para
                this_x_val, _ = normalizer.normalize_one_matrix(
                    x_forecast_set, normalizer.scaler_x,
                    normalizer.outlier_colidx_x, normalizer.threshold, normalizer.method
                )
                # for y, direct normalize to make the result constant across all iterations
                this_y_val, _ = normalizer.normalize_one_matrix(y_val, normalizer.scaler_y)
            # percentiles = [0, 5, 25, 50, 75, 95, 100]
            # print('nn normalized data value quantiles {0}:'.format(percentiles))
            # show_nn_sample_value_quantiles(x_train_set, y_train_set, this_x_val, this_y_val, percentiles)

            do_para_call(
                hybrid_pipeline.train_nn_model, x_train_set, y_train_set,
                x_val=this_x_val, y_val=this_y_val, num_filters=32, batch_size=256, use_callback=False, **params
            )
            cnn_fit_history = hybrid_pipeline.cnn_model.history.history
            excuted_epochs = len(cnn_fit_history['loss'])
            epochs_list.append(excuted_epochs)
            if y_val is not None:
                val_loss_list.append(cnn_fit_history['val_loss'])
            # print('model training process info:')
            # print(pd.DataFrame(hybrid_pipeline.cnn_model.history.history))

            obj_val_index = list(range(len(y_actual)))
            if obj_ratio != 1:
                np.random.shuffle(obj_val_index)
                obj_val_index = obj_val_index[:int(obj_ratio*len(y_actual))]
            # pdb.set_trace()
            y_prediction = hybrid_pipeline.hybrid_make_forecast(
                x_forecast_set[obj_val_index], y_ets_normalizer_set[obj_val_index], value_transformer_set[obj_val_index],
                num_samples=50, bootstrap_sampler='median'
            )

            smape_table = hybrid_pipeline.compute_smape(y_prediction, y_actual[obj_val_index], 0)
            smape_list = list()
            for j in range(num_target):
                smape_list.append(np.mean(smape_table.iloc[:, 2 * j + 1]))

            # mean_smape = np.mean(cnn_fit_history['val_loss'][-3:])
            # mean_smape = cnn_fit_history['val_loss'][-1]

            print('u_smape of val-set: {}; actual epochs excuted: {}; time took: {}'.format(smape_list, excuted_epochs, time.time() - t0))
            return np.mean(smape_list)

        # Optimize everything
        print_section_msg('start searching hyper-para space')
        n_random_starts = np.min([10, n_calls])
        if 'all_random' in kwargs.keys():
            if kwargs['all_random']:
                n_random_starts = n_calls
                print('{} random sample points as initialization'.format(n_calls))
        prev_x, prev_fv = None, None
        if 'history' in kwargs.keys():
            n_random_starts = 0
            prev_x, prev_fv = kwargs['history'].x_iters, kwargs['history'].func_vals
            print('using {} samples as history'.format(len(prev_x)))

        space = Space(normalize_dimensions(para_space))
        n_dims = space.transformed_n_dims
        cov_amplitude = ConstantKernel(1.0, (0.1, 10.0))

        para_l_bound_list = [(0.01, 0.2)] * n_dims
        para_l_bound_list[1] = (0.001, 0.01)
        kernel = RBF(length_scale=np.repeat(0.05, n_dims), length_scale_bounds=para_l_bound_list)
        # kernel = Matern(length_scale=np.ones(n_dims), length_scale_bounds=[(0.01, 3)] * n_dims, nu=1.5)

        kernel = cov_amplitude * kernel + WhiteKernel(noise_level_bounds=(0.1, 0.5))
        estimator = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4, normalize_y=False)
        # estimator.set_params(noise='gaussian')
        res_gp = gp_minimize(
            objective, para_space, n_calls=n_calls, n_random_starts=n_random_starts,
            base_estimator=estimator, acq_func='EI', xi=0.01, x0=prev_x, y0=prev_fv
        )
        # remove this attribute for pickling
        del res_gp.specs['args']['func']
        # add some extra result info to result dict
        res_gp.specs['epochs'] = epochs_list
        res_gp.specs['val_loss'] = val_loss_list
        res_gp.specs['tpratio'] = trapara_ratio_list

        for one_para, one_space in zip(para_space, res_gp.space.dimensions):
            one_space.name = one_para.name

        return res_gp

    def optimize_v1(self, m4_type, n_calls=30):

        hyper_para_space = self.get_hyper_space(m4_type)

        sample_horizon = self.m4_category['horizon'][m4_type[0]]
        season_period = self.m4_category['basea'][m4_type[0]]
        long_season_threshold = self.m4_category['longsea'][m4_type[0]]

        hybrid_pipeline = HybridModelPipelineLocal(sample_horizon, None, season_period, 'log', True,
                                                   'byets', long_season_threshold, None)

        actual_value_id, y_actual, _ = hybrid_pipeline.get_actual_output(self.etsfit_list, sample_horizon, verbose=1)
        y_actual = np.expm1(y_actual)
        print('validation data set shape {0}'.format(y_actual.shape))

        @use_named_args(hyper_para_space)
        def objective(**params):
            t0 = time.time()
            print_section_msg('new iteration seperator')
            print('hyper parameters: {0}'.format(params))

            hybrid_pipeline.set_hybrid_sample_history_length(params['sample_hislen'])

            dropout_rate = (params['dropout_rate'], params['dropout_rate'])
            hybrid_pipeline.hybrid_train_model(
                self.etsfit_list, sample_horizon, params['train_stride'], 0, params['num_train'], True,
                [0], None, 20, 'trim',
                32, params['num_layers'], dropout_rate, 'mae',
                params['learning_rate'], 'adam', 256, params['epochs'],
                verbose_control=0
            )
            excuted_epochs = len(hybrid_pipeline.cnn_model.history.history['loss'])
            # print('model training process info:')
            # print(pd.DataFrame(hybrid_pipeline.cnn_model.history.history))

            id_set, x_forecast_set, y_ets_normalizer_set, value_transformer_set = \
                hybrid_pipeline.get_forecast_input(self.etsfit_list, sample_horizon, verbose=0)

            y_prediction = hybrid_pipeline.hybrid_make_forecast(
                x_forecast_set, y_ets_normalizer_set, value_transformer_set,
                num_samples=50, bootstrap_sampler='median'
            )

            # print(np.array_equal(actual_value_id, id_set))
            smape_table = hybrid_pipeline.compute_smape(y_prediction, y_actual, 0, id_prediction=id_set, id_actual=actual_value_id)
            mean_smape = np.mean(smape_table.smape.values)

            print('u_smape of val-set: {}; actual epochs excuted: {}; time took: {}'.format(mean_smape, excuted_epochs, time.time() - t0))
            return mean_smape

        # Optimize everything
        print_section_msg('start searching hyper-para space')
        res_gp = gp_minimize(
            objective, hyper_para_space, n_calls=n_calls, n_random_starts=5, random_state=0,
            acq_func='EI', xi=0.1, acq_optimizer='lbfgs', n_restarts_optimizer=20, n_jobs=3
        )
        # remove this attribute for pickling
        del res_gp.specs['args']['func']

        return res_gp

import pdb
import time
import numpy as np
import pandas as pd

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Space, Integer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from skopt.utils import normalize_dimensions

from ..cnn.hybrid_model_forecaster import HybridModelForecaster
from ..utils import print_section_msg, do_para_call, show_nn_sample_value_quantiles


class CNNOptimizer:

    num_early_stop = 4
    values_norm_threshold = [20, 50, None]
    values_num_filters = [16, 32, 64, 128, 256]
    # values_num_filters = range(100)     # for rnn exp only: #units para in GRU layer

    def __init__(self, etsfit_list=None):
        self.etsfit_list = etsfit_list

    @staticmethod
    def get_optimal_params(res_gp, **kwargs):
        para_dict = dict()
        if 'para_space' in kwargs.keys():
            para_space = kwargs['para_space']
        else:
            para_space = res_gp.space.dimensions

        # find which iteration in res_gp has the parameter set needed
        sel_para_index = np.argmin(res_gp.func_vals)
        if 'iter_index' in kwargs.keys():
            sel_para_index = kwargs['iter_index']

        para_values = res_gp.x_iters[sel_para_index]
        for one_para, value in zip(para_space, para_values):
            if one_para.name == 'threshold':
                para_dict[one_para.name] = CNNOptimizer.values_norm_threshold[value]
            elif one_para.name == 'num_filters':
                para_dict[one_para.name] = CNNOptimizer.values_num_filters[value]
            else:
                para_dict[one_para.name] = value

        # if the #epoch value is not a search parameter, find it in the extra field of res_gp dict
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
            his_table['threshold'] = [CNNOptimizer.values_norm_threshold[vi] for vi in his_table.threshold]
        his_table['func_val'] = res_gp.func_vals
        his_table = his_table.sort_values('func_val')
        return his_table

    @staticmethod
    def retrieve_tuning_info(res_gp):
        pass

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
    def get_m4_hyper_space(m4_type):
        m4t = m4_type[0]

        m4bd = CNNOptimizer.get_m4_cat_hyper_para_bounds()
        hyper_para_space = list()

        hyper_para_space.append(Integer(10, 30, name='epochs'))
        hyper_para_space.append(Integer(0, 2, name='threshold'))

        num_layer_up = np.ceil(np.log2(m4bd['hislen_h'][m4t]))
        num_layer_up = np.min([num_layer_up, m4bd['nlayer_h'][m4t]])
        hyper_para_space.append(Integer(2, num_layer_up, name='num_layers'))
        hyper_para_space.append(Integer(0, 4, name='num_filters'))
        if m4_type not in ['Yearly', 'Weekly', 'Hourly']:
            hyper_para_space.append(Integer(3, 19, name='num_train'))
        hyper_para_space.append(Integer(m4bd['hislen_l'][m4t], m4bd['hislen_h'][m4t], name='sample_hislen'))

        return hyper_para_space

    @staticmethod
    def get_bac_hyper_space():
        hyper_para_space = list()
        hyper_para_space.append(Integer(10, 50, name='epochs'))
        hyper_para_space.append(Integer(0, 2, name='threshold'))
        hyper_para_space.append(Integer(0, 4, name='num_filters'))
        hyper_para_space.append(Integer(2, 6, name='num_layers'))
        hyper_para_space.append(Integer(9, 19, name='num_train'))
        hyper_para_space.append(Integer(21, 126, name='sample_hislen'))

        return hyper_para_space

    @staticmethod
    def optimize(para_space, hybrid_pipeline, etsfit_list, y_actual_suit, n_calls=10, y_val=None, train_stride=None,
                 holdout_length=0, obj_ratio=1, num_filters=32, use_callbacks=False, model_filename=None, **kwargs):

        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else 0

        actual_value_id, y_actual = y_actual_suit
        batch_size = 256

        num_target = y_actual.shape[-1]
        x_target_colidx = list(range(num_target))
        if hybrid_pipeline.long_season_policy in ['shifbase', 'shifspec']:
            x_target_colidx = list(range(num_target * 2))
        y_target_colidx = list(range(num_target))

        global iter_counter
        iter_counter = 0
        smape_list, val_loss_list, extra_info_list = [], [], []

        @use_named_args(para_space)
        def objective(**params):
            global iter_counter
            t0 = time.time()
            print_section_msg('new iteration seperator')
            print('hyper parameters: {}'.format(params))

            hybrid_pipeline.set_hybrid_sample_history_length(params['sample_hislen'])

            this_train_stride = train_stride
            this_num_filters = num_filters
            if 'train_stride' in params.keys():
                this_train_stride = params['train_stride']
                params.pop('train_stride')
            if 'num_filters' in params.keys():
                this_num_filters = CNNOptimizer.values_num_filters[params['num_filters']]
                params.pop('num_filters')

            x_train_set, y_train_set, _, _ = do_para_call(
                hybrid_pipeline.generate_training_samples, etsfit_list,
                holdout_length=holdout_length, num_val=0, train_stride=this_train_stride, verbose=verbose, **params
            )

            input_shape = x_train_set.shape[1:]
            output_shape = y_train_set.shape[1:]
            cnn_model = HybridModelForecaster()
            do_para_call(cnn_model.build_model, input_shape, output_shape, num_filters=this_num_filters, **params)
            cnn_param_ct = cnn_model.get_param_count()
            print('{} cnn params, {} training samples, ratio {}'.
                  format(cnn_param_ct, len(x_train_set), float(len(x_train_set)) / cnn_param_ct))

            id_set, x_forecast_set, y_ets_normalizer_set, value_transformer_set = hybrid_pipeline.get_forecast_input(
                etsfit_list, holdout_length, verbose=verbose
            )

            # do normalization
            threshold_value = None
            if 'threshold' in params.keys():
                threshold_value = CNNOptimizer.values_norm_threshold[params['threshold']]
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

            if verbose:
                # value check
                percentiles = [0, 5, 25, 50, 75, 95, 100]
                print('nn normalized data value quantiles {0}:'.format(percentiles))
                show_nn_sample_value_quantiles(x_train_set, y_train_set, this_x_val, this_y_val, percentiles)

            epochs = 200
            if 'epochs' in params.keys():
                epochs = params['epochs']
                params.pop('epochs')
            do_para_call(
                hybrid_pipeline.train_nn_model, x_train_set, y_train_set, x_val=this_x_val, y_val=this_y_val,
                num_filters=this_num_filters, batch_size=batch_size, use_callback=use_callbacks,
                epochs=epochs, verbose=verbose, **params
            )
            cnn_fit_history = hybrid_pipeline.cnn_model.history.history
            excuted_epochs = len(cnn_fit_history['loss'])
            if verbose:
                print('model training process info:')
                print(pd.DataFrame(cnn_fit_history))

            if y_val is not None:
                val_loss_list.append(cnn_fit_history['val_loss'])
            if model_filename is not None:
                hybrid_pipeline.serialize(model_filename + '_' + str(iter_counter))

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
            targets_smape = list()
            for j in range(num_target):
                targets_smape.append(np.mean(smape_table.iloc[:, 2 * j + 1]))
            opt_obj = np.mean(targets_smape)

            smape_list.append(targets_smape)
            # add the extra statistics of interest to the list
            elapsed_time = time.time() - t0
            extra_info_list.append([len(x_train_set), cnn_param_ct, excuted_epochs, elapsed_time])
            print('target smape(u/all) of the validation: {}/{}; actual epochs excuted: {}; time took: {}'.
                  format(opt_obj, targets_smape, excuted_epochs, elapsed_time))

            iter_counter += 1
            return np.mean(opt_obj)

        # Optimize everything
        print_section_msg('start searching hyper-para space')
        n_random_starts, prev_x, prev_fv = 10, None, None
        if 'n_random_starts' in kwargs.keys():
            n_random_starts = kwargs['n_random_starts']
        n_random_starts = np.min([n_random_starts, n_calls])
        if 'all_random' in kwargs.keys():
            if kwargs['all_random']:
                n_random_starts = n_calls
                print('{} random sample points as initialization'.format(n_calls))
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
        res_gp.specs['smape'] = smape_list
        res_gp.specs['val_loss'] = val_loss_list
        res_gp.specs['extra_info'] = extra_info_list

        for one_para, one_space in zip(para_space, res_gp.space.dimensions):
            one_space.name = one_para.name

        return res_gp


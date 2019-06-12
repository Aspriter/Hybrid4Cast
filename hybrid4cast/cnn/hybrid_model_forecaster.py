import pickle
import numpy as np
from numbers import Number
from inspect import getfullargspec

import keras.optimizers
import keras.backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json

from . import loss_functions
from .models import WaveNet, StackedGRU
from .custom_layers import ConcreteDropout
from ..utils import sample_concatenate, value_sampler


class HybridModelForecaster:

    def __init__(self):
        self.season_period = None # for experimental use

        self.model = None

        self.loss = None
        self.learning_rate = None
        self.optimizer = None

        self.train_batch_size = None

        self.history = None
        self.is_fitted = False

    @classmethod
    def from_model_file(cls, model_filename):
        '''
        Reconstruct the HybridModelForecaster from model file
        :param model_filename:
        :param train_batch_size:
        :return:
        '''
        forecaster = HybridModelForecaster()
        with open(model_filename + '.json') as f:
            model_json_format = f.read()

        with open(model_filename + '_loss.pkl', 'rb') as f:
            forecaster.loss = pickle.load(f)

        model = model_from_json(model_json_format, {'WaveNet': keras.Model, 'ConcreteDropout': ConcreteDropout})
        model.load_weights(model_filename + '.h5')
        forecaster.model = model
        forecaster.is_fitted = True

        return forecaster

    def store_model(self, model_filename):
        model_json = self.model.to_json()
        # Improtant workaround to unblock deserialization
        with open(model_filename + '.json', 'w') as fid:
            fid.write(model_json)

        with open(model_filename + '_loss.pkl', 'wb') as fid:
            pickle.dump(self.loss, fid, protocol=4)

        # Serialize model weights to HDF5
        self.model.save_weights(model_filename + '.h5')

    def build_model(self, input_shape, output_shape, num_filters=32, num_layers=2, dropout_rate=(0.1, 0.3),
                    loss='mae', learning_rate=1e-3, optimizer='adam', **kwargs):
        # Set up the loss function
        if loss in ['mse', 'mae']:
            pass
        elif loss == 'gaussian':
            output_shape = list(output_shape)
            output_shape[1] *= 2
            output_shape = tuple(output_shape)
        else:
            raise Exception('unrecognized loss type {0}!'.format(loss))
        self.loss = getattr(loss_functions, loss)(output_shape)
        self.learning_rate = learning_rate

        # Neural network model attributes
        # if num_layers is not int, decide it adaptively based on the horizon
        if not isinstance(num_layers, Number):
            num_layers = np.floor(np.log2(output_shape[0]))
            num_layers = np.min([4, num_layers])
            num_layers = int(np.max([2, num_layers]))
        self.model = WaveNet(input_shape, output_shape, num_filters, num_layers, dropout_rate)
        # self.model = StackedGRU(input_shape, output_shape, num_filters, num_layers) # for experiment only

        # Set up optimizer
        self.optimizer = getattr(keras.optimizers, optimizer)(lr=learning_rate)
        allowed_kwargs = self._get_optimizer_args()
        opt_kwargs = dict()
        for key in kwargs.keys():
            if key in allowed_kwargs:
                opt_kwargs[key] = kwargs[key]
                # raise ValueError('{} not a valid optimizer argument.'.format(key))
        self._set_optimizer_args(opt_kwargs)

    def get_param_count(self):
        return self.model.count_params()

    # build and train the model
    def train_model(self, x_train, y_train, x_val=None, y_val=None, batch_size=None, epochs=50,
                    use_callback=False, best_model_temp=None, verbose=0):
        if not self.is_fitted:
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
        if verbose:
            print(self.model.summary())

        if batch_size is None:
            if self.train_batch_size is None:
                # deciding training batch size adaptively
                self.train_batch_size = int(np.max([2 ** (np.ceil(np.log2(len(x_train) / 1e4)) + 3), 64]))
        else:
            self.train_batch_size = batch_size

        val_data = None
        if x_val is not None and y_val is not None:
            val_data = (x_val, y_val)

        callbacks = [TerminateOnNaN()]
        if use_callback:
            if val_data is not None:
                callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=15))
                callbacks.append(ReduceLROnPlateau(
                    monitor='val_loss', factor=0.33, patience=4, verbose=verbose,
                    min_delta=0, min_lr=self.learning_rate/300)
                )

                if best_model_temp is not None:
                    self.store_model(best_model_temp)
                    callbacks.append(ModelCheckpoint(best_model_temp + '.h5', monitor='val_loss', save_best_only=True,
                                                     save_weights_only=True, mode='min'))
            else:
                callbacks.append(EarlyStopping(monitor='loss', min_delta=1e-4, patience=5))
                callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=verbose,
                                                   min_delta=0.001, min_lr=self.learning_rate / 20))

        self.history = self.model.fit(x_train, y_train, batch_size=self.train_batch_size, epochs=epochs,
                                      validation_data=val_data, callbacks=callbacks, verbose=verbose)

        # ???? can spark cache intermediate model files to support use_best param????
        if best_model_temp is not None:
            self.load_model(best_model_temp)

        self.is_fitted = True

    def predict_batch(self, input_x, num_samples=50):
        # Check if model is actually fitted
        if not self.is_fitted:
            raise ValueError('The model has not been fitted.')

        # Repeat the prediction n_samples times to generate samples from
        # approximate posterior predictive distribution.
        block_size = len(input_x)
        bootstrap_x = np.repeat(input_x, [num_samples] * block_size, axis=0)

        # Make predictions for parameters of pdfs then sample from pdfs
        predictions = self.model.predict(bootstrap_x, self.train_batch_size)
        predictions = self.loss.sample(predictions, n_samples=1)

        # Reorganize prediction samples into 3D array
        reshuffled_predictions = []
        for i in range(block_size):
            block = predictions[i * num_samples:(i + 1) * num_samples]
            block = np.expand_dims(block, axis=1)
            reshuffled_predictions.append(block)
        predictions = np.concatenate(reshuffled_predictions, axis=1)

        return predictions

    def predict(self, input_x, batch_size=5000, num_samples=50, bootstrap_sampler='median', verbose=0):
        start, test_size, i = 0, len(input_x), 0
        if verbose:
            print('{0} input samples in total'.format(test_size))

        final_result = []
        while start < test_size:
            end = min(start + batch_size, test_size)
            final_result.append(self.predict_batch(input_x[start:end], num_samples))

            if verbose:
                print('predicting batch {0}:{1} input samples finished...'.format(i, end - start))

            start += batch_size
            i += 1

        final_result = sample_concatenate(final_result, axis=1)
        final_result = np.apply_along_axis(value_sampler, 0, final_result, method=bootstrap_sampler)
        return final_result

    def reset(self):
        """Reset model for refitting."""
        self.is_fitted = False

    def _get_optimizer_args(self):
        """Get optimizer parameters."""
        args = getfullargspec(self.optimizer.__class__)[0]
        args.remove('self')
        return args

    def _set_optimizer_args(self, params):
        """Set optimizer parameters."""
        optimizer_class = self.optimizer.__class__
        optimizer_args = getfullargspec(optimizer_class)[0]
        for key, value in params.items():
            if key in optimizer_args:
                setattr(self.optimizer, key, K.variable(value=value, name=key))

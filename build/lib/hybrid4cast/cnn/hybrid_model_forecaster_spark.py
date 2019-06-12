'''
    Deep4cast with Horovod

    This code runs on DataBricks cluster type 5.0 ML Beta (GPU).

    This code doesn't run on Windows platform, because of the missing depdences required by Horovod (such as MIPS)

'''
import numpy as np
from inspect import getfullargspec

import keras.optimizers
import keras.backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json

import horovod.keras as hvd

from cnn import custom_layers, loss_functions
from cnn.models import WaveNet


'''
HybridModelForecasterSpark: 
    it is HybridModelForecaster built on top of Horovod, that can leverage Spark setup to 
    train the Tensorflow/Keras model in a cluster.

Usage:
    HorovodRunner imported from sparkdl is used to train the Keras/Tensorflow model. for an
    instance:   

    def run_training() :

        # Initalizae horovod for distributed training
        hvd.init()

        # Config tensorflow parameters for GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        sess = tf.Session(config=config)
        K.set_session(sess)

        #
        # Prepare training and test data at here
        # ......

        spark_hybrid_model = HybridModelForecaster()
        spark_hybrid_model.build_model()
        spark_hybrid_model.train_model(x_train, y_train, x_val, y_val)

        # Model training is done here

    #
    # DataBricks provides the detail reference about using HorovodRunner
    # https://databricks.com/blog/2018/11/19/introducing-horovodrunner-for-distributed-deep-learning-training.html
    # API interface
    # https://databricks.github.io/spark-deep-learning/docs/_site/api/python/index.html#sparkdl.HorovodRunner

    from sparkdl import HorovodRunner
    # Assume there are 20 parallel processors (aka. spark task)
    hr = HorovodRunner(np=20)   
    hr.run(run_training_horovod


'''
class HybridModelForecasterSpark:

    def __init__(self):
        self.model = None

        self.loss = None
        self.learning_rate = None
        self.optimizer = None

        self.train_batch_size = None

        self.history = None
        self.is_fitted = False

    @classmethod
    def from_model_file(cls, model_file, train_batch_size=64):
        '''
        Reconstruct the HybridModelForecaster from model file
        :param model_file:
        :param train_batch_size:
        :return:
        '''
        forecaster = HybridModelForecaster()
        forecaster.load_model(model_file)

        forecaster.train_batch_size = train_batch_size
        forecaster.is_fitted = True
        return forecaster

    def build_model(self, input_shape, output_shape, num_filters=32, num_layers=2,
                    loss='mse', learning_rate=1e-3, optimizer='adam', **kwargs):
        # Set up the loss function
        if loss == 'mse':
            pass
        elif loss == 'gaussian':
            output_shape[1] *= 2
        else:
            raise Exception('unrecognized loss type {0}!'.format(loss))
        self.loss = getattr(loss_functions, loss)(output_shape)
        self.learning_rate = learning_rate * hvd.size()

        # Neural network model attributes
        self.model = WaveNet(input_shape, output_shape, filters=num_filters, num_layers=num_layers)

        # Set up optimizer
        self.optimizer = getattr(keras.optimizers, optimizer)(lr=learning_rate)
        allowed_kwargs = self._get_optimizer_args()
        for key, value in kwargs.items():
            if key not in allowed_kwargs:
                raise ValueError('{} not a valid optimizer argument.'.format(key))
        self._set_optimizer_args(kwargs)

    # build and train the model
    def train_model(self, x_train, y_train, x_val=None, y_val=None, epochs=50, use_best=False, verbose=0):
        if not self.is_fitted:
            self.model.compile(loss=self.loss, optimizer=hvd.DistributedOptimizer(self.optimizer))
        if verbose:
            print(self.model.summary())

        # deciding training batch size adaptively
        self.train_batch_size = int(np.max([2 ** (np.floor(np.log2(len(x_train) / 1e4)) + 2), 64]))

        val_data = None
        if x_val is not None and y_val is not None:
            val_data = (x_val, y_val)

        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),

            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),

            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

            TerminateOnNaN()]

        best_mode_filename = None
        if val_data is not None:
            callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=5))
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=verbose,
                                               min_delta=0.003, min_lr=self.learning_rate/20))

            # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
            if use_best and hvd.rank() == 0:
                best_mode_filename = 'bestmodeltemp'
                self.store_model(best_mode_filename)
                callbacks.append(ModelCheckpoint(best_mode_filename + '.h5', monitor='val_loss', save_best_only=True,
                                                 save_weights_only=True, mode='min'))

        self.history = self.model.fit(x_train, y_train, batch_size=self.train_batch_size, epochs=epochs,
                                      validation_data=val_data, callbacks=callbacks, verbose=verbose)

        # ???? can spark cache intermediate model files to support use_best param????
        if use_best:
            self.load_model(best_mode_filename)

        self.is_fitted = True

    def predict(self, test_x, num_samples=100):
        # Check if model is actually fitted
        if not self.is_fitted:
            raise ValueError('The model has not been fitted.')

        # Repeat the prediction n_samples times to generate samples from
        # approximate posterior predictive distribution.
        block_size = len(test_x)
        bootstrap_x = np.repeat(test_x, [num_samples] * block_size, axis=0)

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

    def reset(self):
        """Reset model for refitting."""
        self.is_fitted = False

    def store_model(self, model_filename):

        # Horovod: save model on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            model_json = self.model.to_json()
            with open(model_filename + '.json', 'w') as fid:
                fid.write(model_json)

            # Serialize model weights to HDF5
            self.model.save_weights(model_filename + '.h5')

    def load_model(self, model_filename):
        json_file = open(model_filename + '.json')
        json = json_file.read()
        json_file.close()

        model = model_from_json(json, {'WaveNet': keras.Model, 'ConcreteDropout': custom_layers.ConcreteDropout})
        model.load_weights(model_filename + '.h5')
        self.model = model
        self.is_fitted = True

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

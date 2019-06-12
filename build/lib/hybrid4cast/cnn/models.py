"""Models module.

This module provides access to neural network topologies that can be used
inside forecasters.

"""

import numpy as np
import keras.layers
from keras.models import Model

from . import custom_layers


class StackedGRU(Model):
    """Extends keras.models.Model object.

    Implementation of stacked GRU topology for multivariate time series.

    :param input_shape: Shape of input example.
    :type input_shape: tuple
    :param output_shape: Shape of output consistent with loss function.
    :type output_shape: tuple
    :param units: Number of hidden units for each layer.
    :type units: int
    :param num_layers: Number of stacked layers.
    :type num_layers: int
    :param activation: Activation function.
    :type activation: string

    """

    def __init__(self, input_shape, output_shape, units=32, num_layers=1, activation='relu'):
        """Initialize attributes."""
        self.units = units
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_layer = custom_layers.ConcreteDropout

        # Initialize model and super class
        self.build_layers(input_shape, output_shape)

    def build_layers(self, input_shape, output_shape):
        """Build layers of the network.

        :param input_shape: Length and dimensionality of time series.
        :type input_shape: tuple
        :param output_shape: Output shape for predictions.
        :type output_shape: tuple

        """
        inputs, outputs = self.build_input(input_shape)

        # Core of the network is created here
        for power in range(1, self.num_layers):
            outputs = self.build_gru_block(outputs)

        # Finally we need to match the output dimensions
        outputs = self.build_output(outputs, output_shape)

        super(StackedGRU, self).__init__(inputs, outputs)

    def build_input(self, input_shape):
        """Return first layer of network."""
        inputs = keras.layers.Input(shape=input_shape)
        outputs_first = self.dropout_layer(temporal=True)(inputs)
        outputs = keras.layers.GRU(
            units=self.units,
            activation=self.activation
        )(outputs_first)

        skip = keras.layers.SeparableConv1D(
            filters=self.units,
            kernel_size=1,
            padding='same',
            name='skip_1',
            use_bias=True
        )(outputs_first)
        outputs = keras.layers.Add()([outputs, skip])

        return inputs, outputs

    def build_output(self, x, output_shape):
        """Return last layer for network."""
        x = self.dropout_layer(temporal=True)(x)
        outputs = keras.layers.Conv1D(
            filters=output_shape[1],
            kernel_size=1,
            padding='same',
            name='skip_out',
            use_bias=True
        )(x)
        outputs = keras.layers.Flatten()(outputs)
        outputs = self.dropout_layer()(outputs)
        outputs = keras.layers.Dense(units=np.prod(output_shape))(outputs)
        outputs = keras.layers.Reshape(target_shape=output_shape)(outputs)

        return outputs

    def build_gru_block(self, x):
        """Build core of the network."""
        x = self.dropout_layer(temporal=True)(x)
        outputs = keras.layers.GRU(
            units=self.units,
            activation=self.activation
        )(x)
        outputs = keras.layers.Add()([outputs, x])

        return outputs


class WaveNet(Model):
    """Extends keras.models.Model object.

    Implementation of WaveNet-like topology for multivariate time series. This 
    architecture is built on the idea of causal convolutions that can
    extract features from time series.

    :param input_shape: Shape of input example.
    :type input_shape: tuple
    :param output_shape: Shape of output consistent with loss function.
    :type output_shape: tuple
    :param filters: Number of hidden units for each layer.
    :type filters: int
    :param num_layers: Number of stacked layers.
    :type num_layers: int
    :param activation: Activation function.
    :type activation: string

    """

    def __init__(self, input_shape, output_shape, filters=32, num_layers=1, dropout_rate=(0.1, 0.3),
                 activation='relu', *args, **kwargs):
        """Initialize attributes."""
        self.filters = filters
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_layer = custom_layers.ConcreteDropout

        # Initialize model and super class
        self.build_layers(input_shape, output_shape)

    def build_layers(self, input_shape, output_shape):
        """Build layers of the network.

        :param input_shape: Length and dimensionality of time series.
        :type input_shape: tuple
        :param output_shape: Output shape for predictions.
        :type output_shape: tuple

        """
        # First layer behaves differently cause of the difference in
        # channels for the conv layers.
        inputs, outputs = self.build_input(input_shape)

        # Core of the network is created here
        for power in range(1, self.num_layers):
            outputs = self.build_wavenet_block(outputs, power)

        # Finally we need to match the output dimensions
        outputs = self.build_output(outputs, output_shape)

        # After layers have been build, the super class needs to initialized
        super(WaveNet, self).__init__(inputs, outputs)

    def build_input(self, input_shape):
        """Return first layer of network."""
        inputs = keras.layers.Input(shape=input_shape)
        outputs_first = self.dropout_layer(temporal=True, init_range=self.dropout_rate)(inputs)
        outputs = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=1,
            use_bias=True,
            name='dilated_1',
            activation=self.activation
        )(outputs_first)

        skip = keras.layers.SeparableConv1D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            name='skip_1',
            use_bias=True
        )(outputs_first)
        outputs = keras.layers.Add()([outputs, skip])

        return inputs, outputs

    def build_output(self, x, output_shape):
        """Return last layer for network."""
        x = self.dropout_layer(temporal=True, init_range=self.dropout_rate)(x)
        outputs = keras.layers.Conv1D(
            filters=output_shape[1],
            kernel_size=1,
            padding='same',
            name='skip_out',
            use_bias=True
        )(x)
        outputs = keras.layers.Flatten()(outputs)
        outputs = self.dropout_layer()(outputs)
        outputs = keras.layers.Dense(units=np.prod(output_shape))(outputs)
        outputs = keras.layers.Reshape(target_shape=output_shape)(outputs)

        return outputs

    def build_wavenet_block(self, x, power):
        """Build core of the network."""
        x = self.dropout_layer(temporal=True, init_range=self.dropout_rate)(x)
        outputs = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=2 ** power,
            use_bias=True,
            name='dilated_%d' % (2 ** power),
            activation=self.activation
        )(x)
        outputs = keras.layers.Add()([outputs, x])

        return outputs

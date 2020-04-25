from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import Layer
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence


class Sampling(Layer):
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        z_mean, z_log_sigma = inputs
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0., stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        return output_shape


class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        # self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        z_mean, z_log_sigma = inputs

        kl_loss = 1 + 2 * z_log_sigma - K.square(z_mean) - K.exp(2 * z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return inputs

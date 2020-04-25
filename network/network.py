import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Activation, \
    Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Reshape

from base.base_model import BaseModel
from tensorflow.keras.optimizers import Adam
from utils.layers import Sampling, KLDivergenceLayer
from scipy.interpolate import interp1d


class Encoder(BaseModel):
    def define_model(self, model_name):
        input_data = Input(self.config.data.shape, name=f"{model_name}_input_img")
        x = Conv2D(filters=4, kernel_size=3, strides=2, padding="same")(input_data)
        x = LeakyReLU()(x)
        x = Conv2D(filters=8, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        z_mean = Dense(self.config.data.latent_dim)(x)
        z_log_sigma = Dense(self.config.data.latent_dim)(x)
        z_mean, z_log_sigma = KLDivergenceLayer()([z_mean, z_log_sigma])
        model = Model(input_data, [z_mean, z_log_sigma], name=model_name)
        return model

    def build_model(self, model_name):
        raise NotImplementedError("We dont have to build the model")


class Decoder(BaseModel):
    def define_model(self, model_name):
        z_input = Input((self.config.data.latent_dim,), name=f"{model_name}_input_img")
        x = Dense(126)(z_input)
        x = LeakyReLU()(x)
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Reshape((4, 4, 256 // (4 * 4)))(x)
        x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, padding="valid")(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=3, kernel_size=4, strides=1, padding="valid")(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=1, kernel_size=3, strides=1, padding="valid")(x)
        x = LeakyReLU()(x)
        _data_gen = Conv2D(filters=1, kernel_size=1, padding="same")(x)
        data_gen = Activation("sigmoid")(_data_gen)
        model = Model(z_input, data_gen, name=model_name)
        return model

    def build_model(self, model_name):
        raise NotImplementedError("We dont have to build the model")


class VAE(BaseModel):
    def __init__(self, config, encoder, decoder):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder

    def define_model(self, model_name):
        input_data = Input(self.config.data.shape, name=f"{model_name}_input_img")
        z_mean, z_log_sigma = self.encoder(input_data)
        z = Sampling()([z_mean, z_log_sigma])
        img_gen = self.decoder(z)
        model = Model(input_data, img_gen, name=model_name)
        return model

    def build_model(self, model_name):
        combined_model = self.define_model(model_name=model_name)
        optimizer = Adam(
            self.config.model.hyperparameters.lr,
            self.config.model.hyperparameters.beta1,
            self.config.model.hyperparameters.beta2,
            clipvalue=self.config.model.hyperparameters.clipvalue,
            clipnorm=self.config.model.hyperparameters.clipnorm)
        combined_model.compile(optimizer=optimizer, loss=["mse"], loss_weights=[1])
        return combined_model

    def visualize_latent_interpolation(self, base_samples, use_image_prediction=True):
        # display a 2D manifold of the digits
        n = 30  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # we will sample n points within [-15, 15] standard deviations
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)

        digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        for i, yi in enumerate(grid_x):
            start, end = np.random.choice(digits, 2, replace=False)
            image1 = base_samples[start]
            image1 = image1[np.newaxis, ...]
            image2 = base_samples[end]
            image2 = image2[np.newaxis, ...]
            predicted_vector_start = self.encoder.predict(image1)[0]
            predicted_vector_end = self.encoder.predict(image2)[0]
            linfit = interp1d([0, n], np.vstack([predicted_vector_start, predicted_vector_end]), axis=0)

            for j, xi in enumerate(grid_y):
                if use_image_prediction:
                    z_sample = linfit(j)
                    z_sample = z_sample[np.newaxis, ...]
                else:
                    z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        return figure

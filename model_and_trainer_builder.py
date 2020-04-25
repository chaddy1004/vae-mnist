from network.network import VAE, Encoder, Decoder
from trainer.trainer import Trainer


def get_encoder_model_builder(config):
    return Encoder(config)


def get_decoder_model_builder(config):
    return Decoder(config)


def build_model_and_trainer(config, data_loader):
    encoder_builder = get_encoder_model_builder(config)
    decoder_builder = get_decoder_model_builder(config)

    encoder = encoder_builder.define_model(model_name="encoder")
    decoder = decoder_builder.define_model(model_name="decoder")
    vae = VAE(config=config, encoder=encoder, decoder=decoder)
    full_model = vae.build_model(model_name="vae")
    trainer = Trainer(data_loader=data_loader, config=config, model=full_model, vae_model=vae)
    return full_model, trainer


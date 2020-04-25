from utils.latent_space_utils import embedding_dump, create_embeddings, create_sprite, create_metadata
from utils.process_config import process_config
import os
from dataloader.dataloader import Dataloader, MNISTDataloader
from tensorflow.python.keras.models import load_model
from tensorflow.keras import Model
from utils.layers import KLDivergenceLayer, Sampling
import argparse
import tensorflow as tf


def load_keras_model(saved_model_path, custom_objects={}):
    # add custom layer to custom_objects
    return load_model(
        saved_model_path,
        custom_objects=custom_objects
    )


def main(config_file: str):
    custom_objects = {'KLDivergenceLayer': KLDivergenceLayer, "Sampling":Sampling}
    config = process_config(config_file)
    filename = os.path.join(config.exp.saved_model_dir, "model.hdf5")
    vae = load_keras_model(saved_model_path=filename, custom_objects=custom_objects)
    _encoder = vae.get_layer("encoder")
    sampling = vae.get_layer("sampling")(_encoder.output)
    encoder = Model(_encoder.input, sampling)
    os.makedirs(config.exp.embedding_dir, exist_ok=True)
    if config.exp.isMnist:
        data_loader = MNISTDataloader(config=config, ratio=0.03)
    else:
        data_loader = Dataloader(config=config)
    embedding_array, data_array, label_list = create_embeddings(data_loader=data_loader, encoder=encoder, embeddings_dir=config.exp.embedding_dir)
    create_sprite(data_array=data_array, embeddings_dir=config.exp.embedding_dir)
    create_metadata(label_list=label_list, embeddings_dir=config.exp.embedding_dir)
    embedding_dump(embeddings=embedding_array, embeddings_dir=config.exp.embedding_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"])


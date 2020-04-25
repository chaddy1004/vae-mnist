import argparse
from dataloader.dataloader import Dataloader, MNISTDataloader
from model_and_trainer_builder import build_model_and_trainer
from utils.process_config import process_config
import tensorflow as tf
import os
from matplotlib import pyplot as plt


def visualize_model(config, model):
    filename = os.path.join(config.exp.saved_model_dir, f'model_visualization.png')
    tf.keras.utils.plot_model(
        model, to_file=filename, show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )


def main(config_file: str):
    tf.random.set_seed(19971124)
    config = process_config(config_file)
    data_loader = Dataloader(config=config, ratio=0.1)
    print(data_loader.train_size)
    vae, trainer = build_model_and_trainer(config=config, data_loader=data_loader)
    vae.summary()
    visualize_model(config, vae) 
    print("#########################################################")
    print("#########################################################")

    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml", help="config path to use")
    args = vars(ap.parse_args())
    main(config_file=args["config"])


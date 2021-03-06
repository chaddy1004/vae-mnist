import yaml
from dotmap import DotMap
from glob import glob
import os


def get_config_from_yml(yml_file):
    """
    Get the config from a yml file
    :param yml_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(yml_file, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(yml_file):
    config, _ = get_config_from_yml(yml_file)
    config.data.n_classes = 9
    # config.data.img_shape = (config.data.img_size, config.data.img_size, config.data.img_channels)
    if config.exp.isMnist:
        config.data.shape = (28, 28, 1)
    else:
        config.data.shape = (config.data.timesteps, config.data.feature_size)
    exp_dir = os.path.join(config.exp.experiment_dir, config.exp.name)  # where data for each experiment will be saved

    # Directory where tensorboard scalars will be logged
    config.exp.log_dir = os.path.join(exp_dir, "logs")  # where tensorboard scalars will be logged
    os.makedirs(config.exp.log_dir, exist_ok=True)
    # Directory where models will be saved
    config.exp.saved_model_dir = os.path.join(exp_dir, "saved_model")
    os.makedirs(config.exp.saved_model_dir, exist_ok=True)
    config.exp.embedding_dir = os.path.join(exp_dir, "embeddings")

    return config

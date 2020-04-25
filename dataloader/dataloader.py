from glob import glob

import numpy as np
import tensorflow as tf
from seglearn.transform import PadTrunc

from base.base_dataloader import BaseDataLoader
from tensorflow.keras.datasets.mnist import load_data


class Dataloader(BaseDataLoader):
    def __init__(self, config, ratio=1):
        super().__init__(config)
        self.train_ds = None
        self.test_ds = None
        self.embedding_ds = None
        # initialize datasets
        (x_train, y_train), (x_test, _) = load_data()
        full_size = x_train.shape[0]
        data_size = int(full_size * ratio)
        x_train = x_train[0:data_size, ...]
        x_train = x_train[..., np.newaxis]

        x_test = x_test[..., np.newaxis]
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255
        self.y_train = y_train[0:data_size, ...]
        self.train_size = data_size
        self.init_train_ds()
        self.init_test_ds()
        self.init_embedding_ds()
        self.base_samples = []
        for digit in range(10):
            self.base_samples.append(self.x_train[np.nonzero(y_train[0:data_size, ...] == digit)][0])

    def init_train_ds(self):
        self.train_ds = tf.data.Dataset.from_tensor_slices(self.x_train)
        self.train_ds = self.train_ds.shuffle(self.train_size)
        self.train_ds = self.train_ds.batch(batch_size=self.config.trainer.train_batch_size)
        self.train_ds = self.train_ds.prefetch(1)
        return

    def init_embedding_ds(self):
        img_ds = tf.data.Dataset.from_tensor_slices(self.x_train)
        label_ds = tf.data.Dataset.from_tensor_slices(self.y_train)
        self.embedding_ds = tf.data.Dataset.zip((img_ds, label_ds))
        self.embedding_ds = self.embedding_ds.shuffle(self.train_size)
        self.embedding_ds = self.embedding_ds.batch(batch_size=self.config.trainer.train_batch_size)
        self.embedding_ds = self.embedding_ds.prefetch(1)
        return

    def init_test_ds(self):
        self.test_ds = tf.data.Dataset.from_tensor_slices(self.x_test)
        self.test_ds = self.train_ds.shuffle(self.train_size)
        self.test_ds = self.train_ds.batch(batch_size=self.config.trainer.test_batch_size)
        self.test_ds = self.train_ds.prefetch(1)
        return

    def get_train_data_generator(self):
        return self.train_ds

    def get_train_data_size(self):
        return self.train_size

    def get_embedding_data_generator(self):
        return self.embedding_ds

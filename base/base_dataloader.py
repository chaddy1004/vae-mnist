import abc


class BaseDataLoader:
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def init_train_ds(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_test_ds(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_embedding_ds(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_valid_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data_size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_valid_data_size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_size(self):
        raise NotImplementedError

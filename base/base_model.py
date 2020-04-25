import abc


class BaseModel:
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def define_model(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def define_model(self, **kwargs):
        raise NotImplementedError

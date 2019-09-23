import abc


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, dataset):
        pass

    @abc.abstractmethod
    def predict(self, dataset):
        pass
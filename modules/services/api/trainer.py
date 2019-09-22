import abc


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_employee(self, company_name, id):
        pass

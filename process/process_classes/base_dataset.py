from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_data_row(self):
        """
        This function is used to load the dataset.
        :return: the loaded dataset.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        This function is used to preprocess the dataset (delete useless keys, format transformation, etc.).
        make it qualified for response generation.
        """
        pass

    @abstractmethod
    def save_data(self):
        """
        This function is used to save the dataset.
        """
        pass

import numpy as np

from abc import ABC, abstractmethod
from tensorflow import keras


class BaseModel(ABC):
    def __init__(self, x_train: np.ndarray, num_classes: int = None):
        self.model = self._make_model(x_train)

    @abstractmethod
    def _make_model(self, x_train: np.ndarray, num_classes: int = None) -> keras.Model:
        pass

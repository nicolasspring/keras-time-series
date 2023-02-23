import typing as t
from abc import ABC, abstractmethod

import numpy as np


class BaseDataLoader(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _get_train_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _get_test_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _get_valid_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _get_classes(self) -> np.ndarray:
        pass

    def _shuffle(self):
        self.idx = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[self.idx]
        self.y_train = self.y_train[self.idx]

    def get_train_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self._get_train_set()

    def get_test_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self._get_test_set()

    def get_valid_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self._get_valid_set()

    def get_classes(self) -> np.ndarray:
        return self._get_classes()

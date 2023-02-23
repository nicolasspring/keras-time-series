import typing as t
from abc import ABC, abstractmethod

import numpy as np


class BaseDataLoader(ABC):
    def __init__(self) -> None:
        self.x_train: int
        self.y_train: int
        self.x_test: int
        self.y_test: int
        self.x_valid: int
        self.y_valid: int

    @abstractmethod
    def _get_train_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _get_test_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _get_valid_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        pass

    def get_train_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self._get_train_set()

    def get_test_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self._get_test_set()

    def get_valid_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self._get_valid_set()

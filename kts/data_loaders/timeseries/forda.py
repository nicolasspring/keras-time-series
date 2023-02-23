import typing as t

import numpy as np

from kts.data_loaders.timeseries.base import BaseDataLoader


class FordADataLoader(BaseDataLoader):
    """
    data loader for the FordA dataset. heavily inspired by the Keras tutorial:
    https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    """

    def __init__(self) -> None:
        super().__init__()
        self.root_url = (
            "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        )
        self.x_train, self.y_train = self.read_ford_a(f"{self.root_url}FordA_TRAIN.tsv")
        self.x_test, self.y_test = self.read_ford_a(f"{self.root_url}FordA_TEST.tsv")

    def read_ford_a(self, filename: str) -> t.Tuple[np.ndarray, np.ndarray]:
        self.data = np.loadtxt(filename)
        y = self.data[:, 0]
        x = self.data[:, 1:]
        return x, y.astype(int)

    def _get_train_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def _get_test_set(self) -> t.Tuple[np.ndarray, np.ndarray]:
        self.x_test, self.y_test

    def _get_valid_set(self):
        raise NotImplementedError("FordA dataset does not contain a validation set")

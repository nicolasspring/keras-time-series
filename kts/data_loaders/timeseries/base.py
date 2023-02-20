import typing as t


class BaseDataLoader:
    def __init__(self) -> None:
        pass

    def get_train_set(self):
        pass

    def get_test_set(self):
        pass

from kts.data_loaders.timeseries.base import BaseDataLoader


class FordADataLoader(BaseDataLoader):
    def __init__(self) -> None:
        super().__init__()
        self.root_url = (
            "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        )

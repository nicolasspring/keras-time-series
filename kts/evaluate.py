import argparse
from pathlib import Path

from tensorflow import keras

from kts import constants as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-loader",
        type=str,
        required=True,
        choices=C.DATA_LOADERS.keys(),
        help="data loader to use",
        metavar="STR",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        help="location of the trainied model on the disk",
        metavar="PATH",
    )
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    data_loader = C.DATA_LOADERS[args.data_loader]()
    model = keras.models.load_model(args.model_file)
    x_test, y_test = data_loader.get_test_set()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)


if __name__ == "__main__":
    args = parse_args()
    main(args)

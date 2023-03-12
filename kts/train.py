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
        "--model",
        type=str,
        required=True,
        choices=C.MODELS.keys(),
        help="model to use",
        metavar="STR",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs for model training",
        metavar="INT",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for model training",
        metavar="INT",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer for model training",
        metavar="STR",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        help="loss function for model training",
        metavar="STR",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="patience for early stopping",
        metavar="INT",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        help="output file for the trained model",
        metavar="PATH",
    )
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    data_loader = C.DATA_LOADERS[args.data_loader]()
    x_train, y_train = data_loader.get_train_set()
    classes = data_loader.get_classes()
    num_classes = len(classes)
    model = C.MODELS[args.model](x_train, num_classes).model

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model_out, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, verbose=1
        ),
    ]
    model.compile(
        optimizer=args.optimizer,
        loss=args.loss,
        metrics=[args.loss],
    )
    _ = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)

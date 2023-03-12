# keras-time-series
Time series analysis playground

## Setup
To create a conda environment with the necessary dependencies, use the following command:
```bash
conda env create -f environment.yml
```

## Model Training
To train a classification model on the FordA dataset, use the following command:
```bash
python3 -m kts.train \
    --data-loader forda \
    --model conv_forda \
    --epochs 500 \
    --batch-size 32 \
    --optimizer "adam" \
    --loss "sparse_categorical_crossentropy" \
    --early-stopping-patience "50" \
    --model-out "./model.forda.h5"

```

## Evaluation
Evaluating a previously trained model:
```bash
python3 -m kts.evaluate \
    --data-loader forda \
    --model-file "./model.forda.h5"
```

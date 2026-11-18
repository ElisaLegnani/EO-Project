# Satellite Land Classification

A PyTorch-based Earth observation image classification project using the EuroSAT Sentinel-2 RGB dataset.

**Objective**: Train a CNN to classify satellite image patches into ten land-cover categories.

The `data/EuroSAT_RGB/` folder should contain the EuroSAT RGB dataset from the [EuroSAT GitHub repository](https://github.com/phelber/EuroSAT). It contains 27,000 labelled image patches across 10 land-cover classes. Each image is 64 × 64 pixels with three colour channels (RGB).

## Training

Train the satellite classification model:

```
python -m satellite_classifier.train
```

## Experiment tracking

Start the MLflow dashboard:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000

## Model checkpoint

The best model checkpoint is saved to:

```
models/best_model.pth
```

## Generated artifacts

Training curves and the validation confusion matrix are saved to:

```
artifacts/
```
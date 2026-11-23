# Satellite Image Classification

![CI](https://github.com/ElisaLegnani/EO-Project/actions/workflows/tests.yml/badge.svg) 
![Top Language](https://img.shields.io/github/languages/top/ElisaLegnani/EO-Project)

A machine-learning pipeline for classifying satellite image patches into ten land-cover categories using PyTorch and the EuroSAT Sentinel-2 RGB dataset.

The project combines deep learning, experiment tracking, model evaluation and API development to build a reproducible satellite image-classification application.

## Table of contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)

## Project Overview

The objective is to develop a small but reproducible deep-learning application for satellite image classification, covering the complete workflow from data preparation to model deployment.

Given a satellite image patch, a convolutional neural network (CNN) predicts its corresponding land-cover category.

The project covers the following stages:

- **Data exploration and preprocessing**: inspect image dimensions, class distributions and representative samples, then load RGB images, convert them into tensors and apply the necessary transformations.
- **Model development**: implement and train a CNN using PyTorch.
- **Experiment tracking**: record training parameters, metrics and artefacts with MLflow.
- **Model evaluation**: assess performance on a held-out test set using classification metrics.
- **API development**: serve model predictions through a REST API built with FastAPI, Uvicorn and Pydantic.
- **Interactive frontend**: develop a Streamlit web interface for uploading satellite images, visualising predictions and exploring class probabilities.
- **Containerisation**: package the application components using Docker.
- **Testing and code quality**: use Pytest for automated testing and Ruff for linting and formatting, with GitHub Actions to automate the testing and quality checks through continuous integration.

## Dataset

This project uses the [EuroSAT RGB dataset](https://github.com/phelber/EuroSAT), a publicly available satellite image dataset based on Sentinel-2 imagery. It contains 27,000 labelled image patches across 10 land-cover categories. Each image is 64 x 64 pixels with three colour channels (RGB).

The dataset is not included in this repository and should be downloaded separately. After downloading and extracting the dataset, place it in the `data/EuroSAT_RGB/` folder.

### Data inspection

To inspect and visualize the data, run:

```
python -m satellite_classifier.inspect_data
python -m satellite_classifier.visualize_data
```

Here is an example of the satellite image patches and their corresponding land-cover categories:

![EuroSAT sample images](docs/images/eurosat_samples.png)

## Installation

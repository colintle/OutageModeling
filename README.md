# Weather Impacts and Outage Prediction Using Distribution Networks' Topology and Physical Features - Outage Model CLI

This project includes a Command Line Interface (CLI) for preprocessing weather and static data for model training, as well as training and validating a GATRNN model for predicting outages.

## Authors

- Kenneth McDonald
- Colin T. Le
- Zhihua Qu

## Setup

To get the CLI running, first install the package in editable mode:

```bash
pip install --editable .
```

Then, visit the [PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to install the correct PyTorch package for either CPU or CUDA.

## CLI Commands

### 1. Preprocess Data
```
outage-model preprocess [OPTIONS]
```
Preprocess the weather and static data for model training.

**Options:**
- `--node-static-features`: One or more physical node features to be considered for modeling.  
  Example: `--node-static-feature elevation`
- `--edge-static-feature`: One or more physical edge features to be considered for modeling.  
  Example: `--edge-static-feature length`
- `--data-folder PATH`: Input relative path to the folder containing the model data.
- `--weather-features`: One or more weather features (see the list of possible weather events).
- `--output PATH`: Output path to save both the CSV file and the pickle file.

### 2. Train and Validate Model
```
outage-model train-validate [OPTIONS]
```
Train and validate the GATRNN model with the given parameters.

**Options:**
- `--pkl-file PATH`: Input path to the pickle file containing datasets.
- `--epochs INT`: Number of training epochs.
- `--learning-rate FLOAT`: Learning rate for the optimizer.
- `--hidden-size INT`: Hidden size for the model.
- `--validation-scale FLOAT`: Scale for validation data.
- `--output-model-file PATH`: Output path to save the trained model as a `.pth` file.

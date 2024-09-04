import click
import torch
import random
import pickle
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from .util.functions import normalizeDataset, trainValidate
from .util.GATRNN import GATRNN

@click.group(name="train-validate")
def TRAIN_VALIDATE():
    pass

@TRAIN_VALIDATE.command()
@click.option('--pkl-file', type=click.Path(exists=True), default="dataDict.pkl", help='Path to the pickle file containing datasets.')
@click.option('--output-model-file', type=click.Path(), default="gatrnn_model.pth", help='Path to save the trained model.')
@click.option('--epochs', type=int, default=1500, help='Number of training epochs.')
@click.option('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
@click.option('--hidden-size', type=int, default=40, help='Hidden size for the model.')
@click.option('--validation-scale', type=float, default=1.0, help='Scale for validation data.')
def train_validate(pkl_file, output_model_file, epochs, learning_rate, hidden_size, validation_scale):
    """Train and validate the GATRNN model with the given parameters."""

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_model_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pkl_file, "rb") as f:
        datasets = pickle.load(f)

    datasets_t = datasets['train']
    datasets_v = datasets['validate']
    results = datasets["sF"]

    feature_names = {
        'node_static_features': datasets["node_static_features"],
        'edge_static_features': datasets["edge_static_features"],
        'node_dynamic_features': datasets["node_dynamic_features"],
    }

    nDatasets_t = normalizeDataset(datasets_t, results, feature_names)
    nDatasets_v = normalizeDataset(datasets_v, results, feature_names)

    criterion = torch.nn.MSELoss()

    model = GATRNN(
        num_static_node_features=len(datasets["node_static_features"]), 
        num_static_edge_features=len(datasets["edge_static_features"]),
        num_weather_features=len(datasets["node_dynamic_features"]), 
        hidden_size=hidden_size
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, val_loss = trainValidate(
        model, optimizer, criterion, device, nDatasets_t, nDatasets_v, results, 
        epochs=epochs, validation_scale=validation_scale
    )

    torch.save(model.state_dict(), output_model_file)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    TRAIN_VALIDATE()

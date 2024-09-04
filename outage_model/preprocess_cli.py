import click
import torch
import numpy as np
import os
import csv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from .util.functions import findStats

@click.group(name="preprocess")
def PREPROCESS():
    pass

@PREPROCESS.command()
@click.option('--data-folder', type=click.Path(exists=True), required=True, help='Path to the folder containing the model data.')
@click.option('--edge-static-features', type=str, multiple=True, required=True, help='Static features of the edges.')
@click.option('--node-static-features', type=str, multiple=True, required=True, help='Static features of the nodes.')
@click.option('--weather-features', type=click.Choice(['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco'], case_sensitive=True), multiple=True, required=True, help='Weather features to process.')
@click.option('--output', type=click.Path(), required=True, help='Output path to save both the CSV file and pickle file.')
def preprocess(data_folder, edge_static_features, node_static_features, weather_features, output):
    """Preprocess the weather and static data for model training."""

    # Ensure the output directory exists
    if not os.path.exists(output):
        os.makedirs(output)

    dataset_names = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    datasets = []

    for index, dataset in enumerate(dataset_names):
        nL = pd.read_csv(f"{data_folder}/{dataset}/nodeList.csv")
        eL = pd.read_csv(f"{data_folder}/{dataset}/edgeList.csv")

        sourceList = eL['source'].to_numpy()
        targetList = eL['target'].to_numpy()

        edgeList = np.zeros((len(sourceList), 2))
        edgeList[:, 0] = sourceList
        edgeList[:, 1] = targetList

        nodeStaticFeatures = np.zeros((len(nL), len(edge_static_features)))
        edgeStaticFeatures = np.zeros((len(eL), len(node_static_features)))

        for index, feature in enumerate(edge_static_features):
            edgeStaticFeatures[:, index] = eL[feature].to_numpy()

        for index, feature in enumerate(node_static_features):
            nodeStaticFeatures[:, index] = nL[feature].to_numpy()

        targets = nL['Probability'].to_numpy()

        nodeWeatherFeatures = [f for f in os.listdir(os.path.join(data_folder, dataset, "nodes")) if os.path.isfile(os.path.join(data_folder, dataset, "nodes", f))]

        if len(nodeWeatherFeatures) == 0:
            continue
        
        first = pd.read_csv(os.path.join(data_folder, dataset, "nodes", nodeWeatherFeatures[0]))
        nodeDynamicFeatures = np.zeros((len(nL), len(first.columns), len(nodeWeatherFeatures)))

        for index, ts in enumerate(weather_features):
            data = pd.read_csv(os.path.join(data_folder, dataset, "nodes", f"{ts}.csv"))
            nodeDynamicFeatures[:, :, index] = data

        dataset = {
            'scenario': index,
            'edge_index': torch.tensor(edgeList, dtype=torch.long),
            'node_static_features': torch.tensor(nodeStaticFeatures),
            'edge_static_features': torch.tensor(edgeStaticFeatures),
            'node_dynamic_features': torch.tensor(nodeDynamicFeatures),
            'targets': torch.tensor(targets, dtype=torch.float)
        }

        datasets.append(dataset)

    # Split datasets into training and validation sets
    train_datasets, validate_datasets = train_test_split(datasets, test_size=0.2, random_state=42)

    # Aggregate data for statistics
    nodeData = {f: [] for f in node_static_features}
    edgeData = {f: [] for f in edge_static_features}
    weatherData = {f: [] for f in weather_features}
    probabilities = []

    for key in datasets:
        for index, feature in enumerate(node_static_features):
            nodeData[feature] = key['node_static_features'][:, index].numpy()

        for index, feature in enumerate(edge_static_features):
            edgeData[feature] = key['edge_static_features'][:, index].numpy()

        for index, feature in enumerate(weather_features):
            weatherData[feature] = key["node_dynamic_features"][:, :, index].numpy()

        probabilities.append(key['targets'].numpy())

    results = {}

    for feature, data in nodeData.items():
        mean_val, max_val, min_val = findStats(data)
        results[f"meanNode{feature}"] = mean_val
        results[f"rangeNode{feature}"] = max_val - min_val

    for feature, data in edgeData.items():
        mean_val, max_val, min_val = findStats(data)
        results[f"meanEdge{feature}"] = mean_val
        results[f"rangeEdge{feature}"] = max_val - min_val

    for feature, data in weatherData.items():
        mean_val, max_val, min_val = findStats(data)
        results[f"mean{feature}"] = mean_val
        results[f"range{feature}"] = max_val - min_val

    mean_val, max_val, min_val = findStats(probabilities)
    results["meanProb"] = mean_val
    results["rangeProb"] = max_val - min_val

    # Save statistics to CSV
    csv_filename = os.path.join(output, "sF_NE_dict.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, val in results.items():
            writer.writerow([key, val])

    # Save datasets to pickle file
    datasetDict = {
        'train': train_datasets,
        'validate': validate_datasets,
        'sF': results,
        'edge_static_features': edge_static_features,
        'node_static_features': node_static_features,
        'node_dynamic_features': weather_features
    }

    pkl_filename = os.path.join(output, "dataDict.pkl")
    with open(pkl_filename, 'wb') as f:
        pickle.dump(datasetDict, f)

if __name__ == "__main__":
    PREPROCESS()

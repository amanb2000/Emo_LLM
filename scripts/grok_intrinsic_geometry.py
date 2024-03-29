# Grokking LLM Emotional Latent Space in Human Interpretable Dimensions
### Parsing LLM emptional latent space dataset.
### cayden, Aman

# Imports 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import json
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import argparse

# Setup argparse
parser = argparse.ArgumentParser(description='Grokking LLM Emotional Latent Space')
parser.add_argument('--use-pca', action='store_true', help='Enable PCA preprocessing')
parser.add_argument('--pca-components', type=int, default=20, help='Number of components for PCA')
parser.add_argument('--knn-clusters', type=int, default=5, help='Number of clusters for KNN')
args = parser.parse_args()

# Use argparse values
USE_PCA = args.use_pca
PCA_COMPONENTS = args.pca_components
KNN_CLUSTERS = args.knn_clusters

def load_and_split_data(json_file_path, train_ratio=0.6, e1_ratio=0.15, e2_ratio=0.15, e3_ratio=0.1):
    print("Parsing training data JSON...")
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # get all the possible adjectives prompts in the dataset
    all_adjectives = list(set(entry['adjective'] for entry in data))
    print("all_adjectives")
    print(all_adjectives)
    print(len(all_adjectives))
    all_prompts = list(set(entry['prompt_template'] for entry in data))
    print("all_prompts")
    print(all_prompts)
    print(len(all_prompts))

    # get the size of each eval set
    e1_size = int(len(all_prompts) * e1_ratio)
    e2_size = int(len(all_adjectives) * e2_ratio)
    e3_size = int(min(len(all_prompts), len(all_adjectives)) * e3_ratio)
#    e1_size = int(len(data) * e1_ratio)
#    e2_size = int(len(data) * e2_ratio)
#    e3_size = int(len(data) * e3_ratio)

    # get the special adjectives/prompts to hold out for evals
    e1_prompts = random.sample(all_prompts, e1_size)
    e2_adjectives = random.sample(all_adjectives, e2_size)
    e3_adjectives = random.sample([adj for adj in all_adjectives if adj not in e2_adjectives], e3_size)
    e3_prompts = random.sample([prompt for prompt in all_prompts if prompt not in e1_prompts], e3_size)

    # make those eval sets
    E1_set = [entry for entry in data if entry['prompt_template'] in e1_prompts]
    E2_set = [entry for entry in data if entry['adjective'] in e2_adjectives]
    E3_set = [entry for entry in data if entry['adjective'] in e3_adjectives and entry['prompt_template'] in e3_prompts]

    # training data is what's left over
    remaining_data = [entry for entry in data if entry not in E1_set + E2_set + E3_set]
    train_size = int(len(remaining_data) * train_ratio)
    train_set = random.sample(remaining_data, train_size)

    print("--- Data loaded.")

    return train_set, E1_set, E2_set, E3_set

def extract_features_labels(data):
    features = []
    labels = []

    for item in data:
        # Flatten each list of lists of lists in latent_space into a single list for each data entry
        latent_vectors = item['latent_space']
        # Adjust flattening for an extra level of nesting
        flattened_vector = [val for sublist in latent_vectors for subsublist in sublist for val in subsublist]
        features.append(flattened_vector)

        # Label extraction remains the same
        labels.append(1 if item['valence_good'] else 0)

    return np.array(features), np.array(labels)

def preprocess_features(features, mean=None, scale=None, pca_model=None):
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit or apply normalization
    if mean is None:
        normalized_features = scaler.fit_transform(features)
        mean = scaler.mean_
        scale = scaler.scale_
    else:
        scaler.mean_ = mean
        scaler.scale_ = scale
        normalized_features = scaler.transform(features)

    # Apply PCA if enabled
    if USE_PCA:
        if pca_model is None:
            pca_model = PCA(n_components=PCA_COMPONENTS)
            pca_features = pca_model.fit_transform(normalized_features)
        else:
            pca_features = pca_model.transform(normalized_features)
        return pca_features, mean, scale, pca_model
    else:
        return normalized_features, mean, scale, None

def train_lr_classifier(features, labels):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(features, labels)
    return classifier

def test_lr_classifier(classifier, features, labels):
    y_pred = classifier.predict(features)
    print("Classification Report:\n", classification_report(labels, y_pred))

def train_knn_classifier(features, labels, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier with the given features and labels.

    Parameters:
    - features: The feature matrix for training data.
    - labels: The label vector for training data.
    - n_neighbors: The number of neighbors to use for k-nearest neighbors voting.

    Returns:
    - The trained KNN classifier.
    """
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(features, labels)
    return classifier

def test_knn_classifier(classifier, features, labels):
    """
    Test (evaluate) the trained K-Nearest Neighbors classifier on a test dataset.

    Parameters:
    - classifier: The trained KNN classifier.
    - features: The feature matrix for test data.
    - labels: The label vector for test data.

    Prints:
    - Classification report including precision, recall, and F1-score.
    """
    y_pred = classifier.predict(features)
    print("Classification Report:\n", classification_report(labels, y_pred))


# Load training data
json_file_path = '../gpt2_happy_sad_dump.json'
train_set, e1_set, e2_set, e3_set = load_and_split_data(json_file_path)

print(f"Training set size: {len(train_set)}")
print(f"E1 set size: {len(e1_set)}")
print(f"E2 set size: {len(e2_set)}")
print(f"E3 set size: {len(e3_set)}")

# View data
print(item['adjective'] for item in e2_set)
print(item['prompt'] for item in e2_set)
print(item['valence_good'] for item in e2_set)

# Preprocess features (normalize and optionally apply PCA) for the training set
train_features, train_labels = extract_features_labels(train_set)
train_preprocessed_features, mean_norm, scale_norm, pca_model = preprocess_features(train_features)

# Preprocess features for E1 set (if applicable, uncomment and use as needed)
# e1_features, e1_labels = extract_features_labels(e1_set)
# e1_preprocessed_features, _, _, _ = preprocess_features(e1_features, mean_norm, scale_norm, pca_model)

# Preprocess features for E2 set using the same normalization and PCA model
e2_features, e2_labels = extract_features_labels(e2_set)
e2_preprocessed_features, _, _, _ = preprocess_features(e2_features, mean_norm, scale_norm, pca_model)

# Preprocess features for E3 set (if applicable, uncomment and use as needed)
# e3_features, e3_labels = extract_features_labels(e3_set)
# e3_preprocessed_features, _, _, _ = preprocess_features(e3_features, mean_norm, scale_norm, pca_model)

# Train classifier
lr_classifier = train_lr_classifier(train_preprocessed_features, train_labels)
knn_classifier = train_knn_classifier(train_preprocessed_features, train_labels, n_neighbors=KNN_CLUSTERS)

# Assess clasifier on eval sets
print("LR Classifier test on E2:")
test_lr_classifier(lr_classifier, e2_preprocessed_features, e2_labels)
print("KNN Classifier test on E2:")
test_knn_classifier(knn_classifier, e2_preprocessed_features, e2_labels)

# Grokking LLM Emotional Latent Space in Human Interpretable Dimensions
### Parsing LLM emptional latent space dataset.
### cayden, Aman

# Imports 
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
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
import pdb

# Setup argparse
parser = argparse.ArgumentParser(description='Grokking LLM Emotional Latent Space')
parser.add_argument('--dataset-json', action='store', type=str, default='cache/gpt2_happy_sad_03292024.json', help='Path to the dataset JSON file.')
parser.add_argument('--use-pca', action='store_true', help='Enable PCA preprocessing')
parser.add_argument('--plot-lr', action='store_true', help='Plot the linear regression classifier coefficients per layer. Only works if PCA is off!')
parser.add_argument('--plot-all', action='store_true', help='Plot all data as PCA 3D with adjective and valence labels/colors.')
parser.add_argument('--pca-components', type=int, default=20, help='Number of components for PCA')
parser.add_argument('--knn-clusters', type=int, default=5, help='Number of clusters for KNN')
parser.add_argument('--total-layers', type=int, default=12, help='Number of layers of LLM')
parser.add_argument('--output-dir', type=str, default='cache/', help='Path to the cache directory.')

args = parser.parse_args()


# make output-dir if it doesn't exist. Confirm overwrite if it exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    print(f"Output directory {args.output_dir} already exists. Overwrite? (y/n) ")
    if input().lower() != 'y':
        print("Exiting...")
        exit(0)


# Use argparse values
USE_PCA = args.use_pca
if USE_PCA:
    print("Using PCA, thus PLOT_LR will not run.")
PLOT_LR = args.plot_lr
PLOT_ALL_DATA = args.plot_all
PCA_COMPONENTS = args.pca_components
KNN_CLUSTERS = args.knn_clusters
TOTAL_LAYERS = args.total_layers

# open data file
latent_space_data = None
# json_file_path = 'cache/gpt2_happy_sad_03292024.json'
json_file_path = args.dataset_json
print("Loading training data JSON...")

with open(json_file_path, 'r') as file:
    latent_space_data = json.load(file)

def load_and_split_data(data, train_ratio=0.6, e1_ratio=0.15, e2_ratio=0.15, e3_ratio=0.1):
    # get all the possible adjectives prompts in the dataset
    all_adjectives = list(set(entry['adjective'] for entry in data))
    print("all_adjectives")
    print(all_adjectives)
    all_prompts = list(set(entry['prompt_template'] for entry in data))
    print("all_prompts")
    print(all_prompts)

    # get the size of each eval set
    # SLOW PART STARTS HERE>>>
    # e1_size_adj = int(len(all_adjectives) * e1_ratio)
    # e2_size_prompts = int(len(all_prompts) * e2_ratio)
    # e3_size = int(len(data) * e3_ratio)
    # # get the special adjectives/prompts to hold out for evals
    # e1_adjs = random.sample(all_adjectives, e1_size_adj)
    # e2_prompts = random.sample(all_prompts, e2_size_prompts)
    # e3_objs = random.sample(data, e3_size)

    # # make those eval sets
    # E1_set = [entry for entry in data if entry['adjective'] in e1_adjs]
    # E2_set = [entry for entry in data if entry['prompt_template'] in e2_prompts]
    # E3_set = [entry for entry in data if entry in e3_objs]
    # get the size of each eval set
    e1_size_adj = int(len(all_adjectives) * e1_ratio)
    e2_size_prompts = int(len(all_prompts) * e2_ratio)
    e3_size = int(len(data) * e3_ratio)

    # get the special adjectives/prompts to hold out for evals
    e1_adjs = random.sample(all_adjectives, e1_size_adj)
    e2_prompts = random.sample(all_prompts, e2_size_prompts)
    e3_objs = random.sample(data, e3_size)

    # make those eval sets and training set in a single pass
    E1_set = []
    E2_set = []
    E3_set = []
    train_set = []

    for entry in data:
        if entry['adjective'] in e1_adjs:
            E1_set.append(entry)
        elif entry['prompt_template'] in e2_prompts:
            E2_set.append(entry)
        elif entry in e3_objs:
            E3_set.append(entry)
        else:
            train_set.append(entry)

    # training data is what's left over
    # remaining_data = [entry for entry in data if entry not in E1_set + E2_set + E3_set]
    # train_size = int(len(remaining_data) * train_ratio)
    # train_set = random.sample(remaining_data, train_size)
    train_size = int(len(train_set) * train_ratio)
    train_set = random.sample(train_set, train_size)

    print("Initial dataset size: ", len(data))
    print("Training set size: ", len(train_set))
    print("E1 set size: ", len(E1_set))
    print("E2 set size: ", len(E2_set))
    print("E3 set size: ", len(E3_set))

    print("--- Data loaded.")
    # SLOW PART ENDS HERE <<<
    return train_set, E1_set, E2_set, E3_set

def extract_features_labels(data, layers_to_use=None):
    """
    Extract features and labels from the data.

    Parameters:
    - data: The dataset containing latent vectors and labels.
    - layers_to_use: Optional list of integers specifying which transformer layers to use. If None, all layers are used.

    Returns:
    - A tuple of (features, labels), where features is a NumPy array of the flattened selected layers and labels is a NumPy array of the binary labels.
    """
    features = []
    labels = []

    for item in data:
        latent_vectors = item['latent_space']

        if layers_to_use is not None:
            # Filter the latent vectors to only include the specified layers
            latent_vectors = [latent_vectors[i] for i in layers_to_use]

        # Flatten the selected layers into a single list for each data entry
        # Adjust flattening to account for the two levels of nesting now
        flattened_vector = [val for layer in latent_vectors for head in layer for val in head]

        features.append(flattened_vector)
        labels.append(1 if item['class_0_true'] else 0)

    return np.array(features), np.array(labels)

def preprocess_features(features, mean=None, scale=None, pca_model=None, use_pca=False):
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
    if use_pca:
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

def test_lr_classifier(classifier, features, labels, out_path = None):
    y_pred = classifier.predict(features)
    classification_report_ = classification_report(labels, y_pred)
    print("Classification Report:\n", classification_report_)

    if out_path is not None: 
        out_path = os.path.join(out_path)
        with open(out_path, 'w') as f: 
            f.write(classification_report_)

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


import re

def sanitize_filename(filename):
    """
    Sanitizes a string to be safe for use as a filename by removing or replacing characters
    that are not allowed or recommended in Windows and UNIX/Linux filesystems.
    
    Args:
    filename (str): The original filename string to sanitize.
    
    Returns:
    str: A sanitized version of the filename.
    """
    # Remove characters that are invalid for Windows or UNIX/Linux filesystems
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename)
    # Replace leading and trailing periods and spaces (Windows)
    sanitized = re.sub(r'^[. ]+', '', sanitized)
    sanitized = re.sub(r'[. ]+$', '', sanitized)
    # Replace multiple consecutive spaces with a single space
    sanitized = re.sub(r' +', ' ', sanitized)
    # Ensure the filename is not too long
    sanitized = sanitized[:255]
    return sanitized


def test_knn_classifier(classifier, features, labels, out_path = None):
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
    classification_report_ = classification_report(labels, y_pred)
    print("Classification Report:\n", classification_report_)
    if out_path is not None:
        with open(out_path, 'w') as f: 
            f.write(classification_report_)

def plot_3d_pca(data, labels, adjectives=None, output_dir = None, prompt=""):
    # Normalize labels for color scaling
    colors = np.array(labels) - np.min(labels)
    colors = colors / np.max(colors)
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        text=adjectives,  # Use adjectives as markers' text
        mode='markers+text',  # Display both markers and text
        marker=dict(
            size=5,
            color=colors,  # Use normalized labels for color
            opacity=0.8
        )
    )])
    
    # Customize layout
    fig.update_layout(
        title=f'3D PCA Visualization, prompt={prompt}',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )
    
    # Show plot in notebook or export as desired
    fig.show()
    # To export to HTML, uncomment the following line:
    # fig.write_html('3d_pca_visualization.html')
    if output_dir: 
        out_path = os.path.join(output_dir, f"3d_pca_visualization{sanitize_filename(prompt)}.html")
        print("OUTPUT PATH FOR PCA: ", out_path)
        fig.write_html(out_path)


import plotly.graph_objects as go
import numpy as np

def plot_mean_coefficients_per_layer_with_plotly(lr_classifier, layers_to_use, output_dir=None):
    """
    Plot the mean coefficient weights per layer for a trained Logistic Regression classifier using Plotly.

    Parameters:
    - lr_classifier: The trained Logistic Regression classifier.
    - layers_to_use: The layers we trained the classifier on
    """
    coefficients = lr_classifier.coef_.flatten()  # Extract model coefficients
    print("Number of coefficients: {}".format(len(coefficients)))
    total_layers = len(layers_to_use)

    # Assuming equal feature contribution from each layer if not specified
    features_per_layer = len(coefficients) // total_layers

    print("Features per layer: {}".format(features_per_layer))

    # Calculate mean coefficient weight per layer
    mean_coefficients_per_layer = [np.abs(np.mean(coefficients[i*features_per_layer:(i+1)*features_per_layer])) for i in range(total_layers)]

    # Plotting with Plotly
    fig = go.Figure(data=[go.Bar(
        x=[f'Layer {i}' for i in layers_to_use],
        y=mean_coefficients_per_layer,
        marker_color=np.where(np.array(mean_coefficients_per_layer) > 0, 'blue', 'red')  # Color code positive and negative
    )])

    fig.update_layout(
        title='Mean Coefficient Weights per Layer in Logistic Regression Classifier',
        xaxis_title='Layer',
        yaxis_title='Mean Coefficient Value',
        template='plotly_white'
    )

    fig.show()
    # output to disk if cache_dir is specified
    if output_dir:
        print("Saving figure uwu")
        out_path = os.path.join(output_dir, "mean_coefficients_per_layer.html")
        fig.write_html(out_path)
        print("Done -- saved to ", out_path)

# Load training data
train_set, e1_set, e2_set, e3_set = load_and_split_data(latent_space_data)

print(f"Training set size: {len(train_set)}")
print(f"E1 set size: {len(e1_set)}")
print(f"E2 set size: {len(e2_set)}")
print(f"E3 set size: {len(e3_set)}")

# Drop/only use certain layers
#layers_to_use = [0,4,7,9,11] 
layers_to_use = list(range(0,TOTAL_LAYERS))

# Preprocess features (normalize and optionally apply PCA) for the training set
train_features, train_labels = extract_features_labels(train_set, layers_to_use)
train_preprocessed_features, mean_norm, scale_norm, pca_model = preprocess_features(train_features, use_pca=USE_PCA)

# Preprocess features for E1 set (if applicable, uncomment and use as needed)
e1_features, e1_labels = extract_features_labels(e1_set)
e1_preprocessed_features, _, _, _ = preprocess_features(e1_features, mean_norm, scale_norm, pca_model, use_pca=USE_PCA)

# Preprocess features for E2 set using the same normalization and PCA model
e2_features, e2_labels = extract_features_labels(e2_set, layers_to_use)
e2_preprocessed_features, _, _, _ = preprocess_features(e2_features, mean_norm, scale_norm, pca_model, use_pca=USE_PCA)

# Preprocess features for E3 set (if applicable, uncomment and use as needed)
e3_features, e3_labels = extract_features_labels(e3_set)
e3_preprocessed_features, _, _, _ = preprocess_features(e3_features, mean_norm, scale_norm, pca_model, use_pca=USE_PCA)

# Train classifier
lr_classifier = train_lr_classifier(train_preprocessed_features, train_labels)
# save weights of lr_classifier in args.output_dir/weights.npz
np.savez(os.path.join(args.output_dir, "weights.npz"), lr_classifier.coef_, lr_classifier.intercept_)

knn_classifier = train_knn_classifier(train_preprocessed_features, train_labels, n_neighbors=KNN_CLUSTERS)

# Assess clasifier on eval sets
print("LR Classifier test on E1:")
test_lr_classifier(lr_classifier, e1_preprocessed_features, e1_labels, 
                   out_path=os.path.join(args.output_dir, "lr_classifier_eval_e1.txt"))

print("KNN Classifier test on E1:")
test_knn_classifier(knn_classifier, e1_preprocessed_features, e1_labels, 
                    out_path=os.path.join(args.output_dir, "knn_classifier_eval_e1.txt"))

# 
print("LR Classifier test on E2:")
test_lr_classifier(lr_classifier, e2_preprocessed_features, e2_labels, 
                   out_path=os.path.join(args.output_dir, "lr_classifier_eval_e2.txt"))
print("KNN Classifier test on E2:")
test_knn_classifier(knn_classifier, e2_preprocessed_features, e2_labels, 
                    out_path=os.path.join(args.output_dir, "knn_classifier_eval_e2.txt"))

print("LR Classifier test on E3:")
test_lr_classifier(lr_classifier, e3_preprocessed_features, e3_labels, 
                    out_path=os.path.join(args.output_dir, "lr_classifier_eval_e3.txt"))
print("KNN Classifier test on E3:")
test_knn_classifier(knn_classifier, e3_preprocessed_features, e3_labels, 
                    out_path=os.path.join(args.output_dir, "knn_classifier_eval_e3.txt"))

print("Combining all output texts with titles into a main results.txt") 
with open(os.path.join(args.output_dir, "lr_classifier_eval_e1.txt"), 'r') as f: 
    e1_text = "\n=== E1 LINEAR CLASSIFIER EVAL ===\n"
    e1_text += f.read()
with open(os.path.join(args.output_dir, "knn_classifier_eval_e1.txt"), 'r') as f:
    e1_text += "\n=== E1 KNN CLASSIFIER EVAL ===\n"
    e1_text += f.read()
with open(os.path.join(args.output_dir, "lr_classifier_eval_e2.txt"), 'r') as f:
    e2_text = "\n=== E2 LINEAR CLASSIFIER EVAL ===\n"
    e2_text += f.read()
with open(os.path.join(args.output_dir, "knn_classifier_eval_e2.txt"), 'r') as f:
    e2_text += "\n=== E2 KNN CLASSIFIER EVAL ===\n"
    e2_text += f.read()
with open(os.path.join(args.output_dir, "lr_classifier_eval_e3.txt"), 'r') as f:
    e3_text = "\n=== E3 LINEAR CLASSIFIER EVAL ===\n"
    e3_text += f.read()
with open(os.path.join(args.output_dir, "knn_classifier_eval_e3.txt"), 'r') as f:
    e3_text += "\n=== E3 KNN CLASSIFIER EVAL ===\n"
    e3_text += f.read()

use_pca_text = f"=== USE_PCA = {USE_PCA} ===\n"

# write e1_text + e2_text + e3_test to a results.txt in the out dir
with open(os.path.join(args.output_dir, "results.txt"), 'w') as f:
    f.write(use_pca_text + e1_text + e2_text + e3_text)

# View the dataset PCA
all_features, all_labels = extract_features_labels(latent_space_data)
adjectives = [item['adjective'] for item in latent_space_data] # Assuming 'data' is your entire dataset # Optionally, if you want to visualize using specific labels or adjectives
prompts = [item['prompt_template'] for item in latent_space_data] # Assuming 'data' is your entire dataset # Optionally, if you want to visualize using specific labels or adjectives

all_adjectives = list(set(entry['adjective'] for entry in latent_space_data))
all_prompts = list(set(entry['prompt_template'] for entry in latent_space_data))
# pdb.set_trace()


if PLOT_ALL_DATA:
    # all data plot
    all_preprocessed_features, _, _, pca_model = preprocess_features(all_features, use_pca = True)
    plot_3d_pca(all_preprocessed_features, all_labels, adjectives=adjectives, output_dir = args.output_dir, prompt="ALL PROMPTS")
    for prompt in all_prompts: 
        prompt_mask = [item['prompt_template'] == prompt for item in latent_space_data]

        # pdb.set_trace()
        all_features_filtered = all_features[prompt_mask, :]
        all_labels_filtered = all_labels[prompt_mask]
        all_preprocessed_features, _, _, pca_model = preprocess_features(all_features_filtered, use_pca = True)
        # pdb.set_trace()
        plot_3d_pca(all_preprocessed_features, all_labels_filtered, adjectives=np.array(adjectives)[prompt_mask].tolist(), output_dir = args.output_dir, prompt=prompt)

# View the linear regression classifier coefficients per layer (only makes sense to do if we haven't PCA'ed)
if not USE_PCA and PLOT_LR:
    plot_mean_coefficients_per_layer_with_plotly(lr_classifier, layers_to_use, output_dir = args.output_dir)

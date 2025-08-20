# Team Classifier Module

This document explains how ML-Futbol assigns team identities to players using feature embeddings and clustering.

## Purpose

The team classifier assigns a team label (0 or 1) to each detected player. This is important for tactical analysis, distinguishing teammates from opponents, and generating visual overlays.

## How It Works

### Step 1: Player Crops
- Each player bounding box is cropped from the video frame.
### Step 2: Feature Extraction (SigLIP)
- Each crop is passed through a pretrained SigLIP encoder.
- The encoder outputs a high-dimensional feature vector for each player.
### Step 3: Dimensionality Reduction (UMAP)
- The high-dimensional features are reduced to 2D or 3D using UMAP (Uniform Manifold Approximation and Projection).
- This helps simplify the clustering task.
### Step 4: Clustering (KMeans)
- A KMeans model is trained to divide players into 2 clusters: Team 0 and Team 1.
- Team IDs are assigned based on the cluster index.

## Class Reference: `TeamClassifier`

Defined in: `team_classifier.py`

### `__init__(device="cuda")`
- Loads the SigLIP model onto the specified device.
### `extract_features(player_images: list)`
- Takes a list of player image crops.
- Returns a matrix of feature vectors.
### `cluster_features(features: np.ndarray)`
- Reduces dimensions with UMAP.
- Applies KMeans to cluster into 2 groups.
- Returns team ID labels.

## Output

### The classifier generates team IDs which are:
- Saved alongside player coordinates in `player_coordinates.csv`
- Visualized with colored labels in `anottated_player_video.mp4`
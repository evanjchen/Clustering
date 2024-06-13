import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from IPython.display import Image, display

from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
import random as random
import time 
from sklearn.ensemble import RandomForestClassifier




def select_centroids(X,k):
    """Inputs: X: array;     k: number of centroids
    Returns: centroids and labels for points that maximmize the minimum distance from first random centroid"""
    
    # Choose a random starting 
    index = np.random.choice(X.shape[0], 1, replace=False)
    centroids = X[index]
    
    for i in range(k-1):
        print("SELECTING STARTING CENTROIDS ITERATION:", i)
        
         # Compute distance of EVERY point to EACH Centroid  
        distances = np.array([[np.linalg.norm(c - x) for c in centroids] for x in X])

        # For each point, select the centroid that is the MINIUM distance 
        min_distances = np.array([np.min(d) for d in distances])
        
        # select the points that maximmize the minimum distance for the rest of the clusters
        max_dist = np.argmax(min_distances)
        centroids = np.vstack((centroids, X[max_dist]))
    
    distances = np.array([c-X for c in centroids]).reshape(-1, k)
    # distances = np.array([[np.linalg.norm(c - x) for c in centroids] for x in X])
    labels = np.array([np.argmin(d) for d in distances])
    
    return centroids, labels


def kmeans(X, k, centroids=None, max_iter=30, tolerance=0.04):
    """Input: X = array,
            : k = unique points as initial centroids"""

    all_labels = []
    all_centroids = []

    # Instantiate random centroids for first iteration
    if centroids=='kmeans++':
        first_centroid, labels = select_centroids(X,k)
        all_centroids.append(first_centroid)
    else: 
        idx = np.random.choice(X.shape[0], k, replace=False)   # select k unique points from X as initial centroids
        first_centroid = X[idx]
        first_centroid = np.vstack(first_centroid)
        all_centroids.append(first_centroid)
    

    # distances = np.array([c-X for c in centroid]).reshape(-1, k)
    distances = np.array([[np.linalg.norm(c - x) for c in first_centroid] for x in X])
    labels = np.array([np.argmin(d) for d in distances])
    all_labels.append(labels)

    # Find optimal centroids
    for i in range(max_iter):
        print("ITERATION:", i)
        
        centroids = [X[labels==i].mean(axis=0) for i in range(k)]
        centroids = np.vstack(centroids) # vertically stack k centroids
        

        distances = np.array([[np.linalg.norm(c - x) for c in centroids] for x in X])
        labels = np.array([np.argmin(d) for d in distances])

        # Compare with previous centroid
        if i!= 0:

            prev_centroid = all_centroids[-1]
            # Compare the average norm of centroids - previous centroids with the tolerance
            comparison = np.sum((centroids - prev_centroid) / prev_centroid)
            print("COMPARISON:", comparison)
            if np.abs(comparison) < tolerance:
                return centroids, labels
            
            else:
                all_labels.append(labels)
                all_centroids.append(centroids)
    
    return all_centroids[-1], all_labels[-1]

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    data = next(iter(train_loader))[0]
    data = data.numpy()
    data = data.reshape(data.shape[0], -1)
    return data

def compute_knn_distances(data, k, metric):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric).fit(data)
    distances, indices = nbrs.kneighbors(data)
    return distances[:, 1:].mean(axis=1), indices  # Exclude self-distance

def save_outliers_to_csv(mean_distances, filename):
    df = pd.DataFrame({'Index': np.arange(len(mean_distances)), 'Mean Distance': mean_distances})
    df_sorted = df.sort_values(by='Mean Distance', ascending=False).head(50000)
    df_sorted.to_csv(filename, index=False)
    print(f"Saved {filename} with top 50,000 records.")

data = load_cifar10()
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'hamming']  # Different metrics to test
k_value = 10000  # Number of neighbors

for metric in metrics:
    print(f"Processing {metric} metric")
    mean_distances, _ = compute_knn_distances(data, k_value, metric)
    filename = f"outliers_{metric}.csv"
    save_outliers_to_csv(mean_distances, filename)

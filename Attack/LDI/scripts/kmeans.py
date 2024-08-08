import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Ensure the directory exists
if not os.path.exists('/users/home/ygu27/cifar/MIA/data/imgs'):
    os.makedirs('/users/home/ygu27/cifar/MIA/data/imgs')

# Load CSV file
csv_data = pd.read_csv("/users/home/ygu27/cifar/MIA/data/privacy_score/top_index.csv")  # Replace with your CSV file path
indices = csv_data['Index'].tolist()

# Load CIFAR-10 dataset
full_train_dataset = torchvision.datasets.CIFAR10('/users/home/ygu27/cifar/MIA/data/tmp_shadow', train=True, download=True)

# Convert CIFAR-10 data to numpy arrays and extract images and labels
images = np.array([np.array(img).reshape(-1) for img, _ in full_train_dataset])
labels = np.array([label for _, label in full_train_dataset])

# Determine counts of the 5000 indexed samples within each class
unique, counts = np.unique(labels[indices], return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Perform K-means clustering for each class and plot
for i in range(10):
    class_mask = labels == i
    class_images = images[class_mask]
    class_indices = [idx for idx in indices if labels[idx] == i]

    # Number of clusters for this class
    n_clusters = cluster_counts.get(i, 0)
    if n_clusters > 0:
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(class_images)

        # Plot all samples
        plt.figure(figsize=(10, 8))
        plt.scatter(class_images[:, 0], class_images[:, 1], c='blue', label='Other Samples', alpha=0.5)

        # Highlight indexed samples
        indexed_samples = images[class_indices]
        plt.scatter(indexed_samples[:, 0], indexed_samples[:, 1], c='red', label='Indexed Samples', alpha=0.5)

        # Define grid size and create a grid of zeros for counting indexed samples
        grid_size = 5
        block_count = np.zeros((grid_size, grid_size))
        x_min, x_max = min(class_images[:, 0]), max(class_images[:, 0])
        y_min, y_max = min(class_images[:, 1]), max(class_images[:, 1])
        x_step = (x_max - x_min) / grid_size
        y_step = (y_max - y_min) / grid_size

        # Count indexed samples in each block
        for x, y in indexed_samples:
            x_index = min(int((x - x_min) / x_step), grid_size - 1)
            y_index = min(int((y - y_min) / y_step), grid_size - 1)
            block_count[y_index][x_index] += 1

        # Print the block counts for this class
        print(f"Class {i} indexed sample distribution in blocks:")
        for row in block_count:
            print(' '.join('{:4d}'.format(int(number)) for number in row))

        # Add titles and labels
        plt.title(f'Class {i} Clusters')
        plt.legend()

        # Save the figure
        plt.savefig(f'/users/home/ygu27/cifar/MIA/data/imgs/class_{i}_clusters.png')
        plt.close()

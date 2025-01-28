import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import pandas as pd
from PIL import Image

# -----------------------------------------
# Configuration
# -----------------------------------------
DATA_DIR = "~/kekechen_common/datasets/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train"    # Same directory used for training
TRAINING_CSV_PATH = "training_images/training_images.csv"  # CSV with training image paths
MODEL_SAVE_PATH = "models/resnet18_10classes.pth" # Trained model weights
NUM_IMAGES_TO_EVAL = 1000                  # Number of images to evaluate
OUTPUT_CSV = "obs/logit_differences.csv"

os.makedirs(os.path.dirname('./obs'), exist_ok=True)

BATCH_SIZE = 32

# -----------------------------------------
# Step 1: Define a transform and load the full dataset
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the entire folder structure as before
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)  # no transform here for samples extraction

# -----------------------------------------
# Step 2: Identify the training images and exclude them
# -----------------------------------------
train_df = pd.read_csv(TRAINING_CSV_PATH)
train_image_paths = set(train_df["image_path"].apply(os.path.abspath).tolist())

# Build a list of (image_path, label) that are NOT in the training set
test_samples = []
for sample_path, label in full_dataset.samples:
    abs_path = os.path.abspath(sample_path)
    if abs_path not in train_image_paths:
        test_samples.append((sample_path, label))

# Randomly pick 1000 from the available test samples
# Make sure you have at least 1000 images outside the training set
test_samples = random.sample(test_samples, NUM_IMAGES_TO_EVAL)

# -----------------------------------------
# Step 3: Create a custom Dataset for these 1000 images
# -----------------------------------------
class TestDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: List of (img_path, label)
        transform: Any torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return path so we can record it

test_dataset = TestDataset(test_samples, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------------------
# Step 4: Load the trained model
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False)
# Must match how the model was defined during training (10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)

# Load weights
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------------------
# Step 5: Inference & Logit Difference Calculation
# -----------------------------------------
results = []
with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images)  # shape: (batch_size, 10)

        # For each sample in the batch
        for i in range(outputs.size(0)):
            logit_vector = outputs[i]
            # Find top 2 logits using topk
            top2_vals, top2_idx = torch.topk(logit_vector, 2)
            # Difference = best logit - second best logit
            diff = (top2_vals[0] - top2_vals[1]).item()
            results.append([os.path.abspath(paths[i]), diff])

# -----------------------------------------
# Step 6: Save Results to CSV
# -----------------------------------------
df = pd.DataFrame(results, columns=["image_path", "logit_difference"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved logit differences to {OUTPUT_CSV}")

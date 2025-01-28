import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import pandas as pd
from tqdm import tqdm  # NEW

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR = os.path.expanduser(
    "~/kekechen_common/datasets/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train"
)
NUM_CLASSES_TO_TRAIN = 10
BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-3
TRAIN_SPLIT = 0.8

MODEL_SAVE_PATH = "models/resnet18_10classes.pth"
os.makedirs(os.path.dirname('./models'), exist_ok=True)

TRAINING_CSV_PATH = "training_images/training_images.csv"
os.makedirs(os.path.dirname('./training_images'), exist_ok=True)
# ------------------------------------------------------------------------------
# Step 1: Dataset & Transforms
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# ------------------------------------------------------------------------------
# Step 2: Randomly Select 10 Classes & Build a Label Map
# ------------------------------------------------------------------------------
all_class_indices = list(range(len(full_dataset.classes)))
random_class_indices = random.sample(all_class_indices, NUM_CLASSES_TO_TRAIN)
label_map = {old_label: i for i, old_label in enumerate(random_class_indices)}

# ------------------------------------------------------------------------------
# Step 3: Filter Dataset to Keep Only Those 10 Classes
# ------------------------------------------------------------------------------
indices_to_keep = [
    i for i, (_, old_label) in enumerate(full_dataset.samples)
    if old_label in random_class_indices
]
subset_dataset = Subset(full_dataset, indices_to_keep)

# ------------------------------------------------------------------------------
# Step 4: Train/Validation Split
# ------------------------------------------------------------------------------
train_size = int(TRAIN_SPLIT * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_subset, val_subset = random_split(subset_dataset, [train_size, val_size])

# ------------------------------------------------------------------------------
# Step 5: Save Training Image Paths & NEW Labels to CSV
# ------------------------------------------------------------------------------
train_indices_in_subset = train_subset.indices
image_paths, labels = [], []

for idx in train_indices_in_subset:
    real_index = subset_dataset.indices[idx]
    img_path, old_label = full_dataset.samples[real_index]
    new_label = label_map[old_label]
    image_paths.append(os.path.abspath(img_path))
    labels.append(new_label)

df_train = pd.DataFrame({"image_path": image_paths, "label": labels})
df_train.to_csv(TRAINING_CSV_PATH, index=False)
print(f"Training image paths (with new labels) saved to {TRAINING_CSV_PATH}")

# ------------------------------------------------------------------------------
# Step 6: Create a RemappedDataset to Return (image, new_label)
# ------------------------------------------------------------------------------
class RemappedDataset(Dataset):
    """
    Wraps a dataset, converting old_label -> [0..NUM_CLASSES_TO_TRAIN-1].
    """
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, old_label = self.dataset[idx]
        new_label = self.label_map[old_label]
        return img, new_label

train_dataset = RemappedDataset(train_subset, label_map)
val_dataset = RemappedDataset(val_subset, label_map)

# ------------------------------------------------------------------------------
# Step 7: Create DataLoaders
# ------------------------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------------------------------------------------------------------------------
# Step 8: Define Model, Loss, Optimizer
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES_TO_TRAIN)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------------------------
# Step 9: Training Loop with Progress Bar
# ------------------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Create a tqdm progress bar for the training loader
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Update the progress bar with current loss
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

# ------------------------------------------------------------------------------
# Step 10: Save the Trained Model
# ------------------------------------------------------------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Trained model saved to {MODEL_SAVE_PATH}")

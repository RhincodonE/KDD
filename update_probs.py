import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import norm
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18

# ------------------------------------------------------------------------------
# Helper Function: Compute CDF of Logit Gap
# ------------------------------------------------------------------------------
def compute_cdf_of_logit_gap(image_path, model, device, inference_transform, normal_dist):
    """
    1) Load image.
    2) Forward pass through the model to get logits.
    3) Compute logit_gap = top1 - top2.
    4) Compute logit_transform = log(gap / (1 - gap)), with clamping.
    5) Compute CDF = normal_dist.cdf(logit_transform).

    Returns:
        tuple: (logit_gap, logit_transform, cdf_value)
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)[0]  # Shape: [10]

    top2_vals, _ = torch.topk(logits, 2)
    gap = (top2_vals[0] - top2_vals[1]).item()

    # Compute logit transform with clamping
    eps = 1e-6
    gap_clamped = max(eps, min(1 - eps, gap))
    logit_val = np.log(gap_clamped / (1.0 - gap_clamped))

    # Compute CDF
    cdf_val = normal_dist.cdf(logit_val)

    return gap, logit_val, cdf_val

# ------------------------------------------------------------------------------
# Helper Function: Update Hierarchy Probabilities
# ------------------------------------------------------------------------------
def update_probabilities(
    path_list,
    cdf_val,
    hierarchy_prob_df,
    mapping_df,
    hier_cols,
    total_levels=16
):
    """
    Updates the probabilities in hierarchy_prob_df based on the CDF value.

    - The node at path_list[0] (root) is skipped (no updates).
    - For each node from path_list[1] onward (level j), we:
        1) Find siblings at level j, where parent_node == path_list[j-1].
        2) If C > 1 (there are siblings):
           - Increase/decrease the node's probability by delta_node,
             and do the opposite for siblings by delta_sibling.
           - Clamp probabilities to [0,1].
           - Re-normalize so all children at that (level, parent) sum to 1.
    - Delta increments in this example are fixed at:
         delta_node = 1 / (10*C)
         delta_sibling = 1 / (10*C * (C - 1))
      to produce small adjustments.

    Args:
        path_list (list): Hierarchy path (root -> leaf).
        cdf_val (float): Normal distribution CDF for the logit gap.
        hierarchy_prob_df (pd.DataFrame): Has columns [hierarchy_level, parent_node, child_node, probability].
        mapping_df (pd.DataFrame): Not directly used here, but included for consistency.
        hier_cols (list): Also not used directly in this function, but can be relevant for other logic.
        total_levels (int): Unused in this version, but kept for signature consistency.
    """

    # Skip path_list[0] (root), update nodes from path_list[1..].
    for j in range(1, len(path_list)):
        node = path_list[j]
        parent = path_list[j - 1]
        level = j  # If your CSV stores this node at hierarchy_level=j

        # Filter the dataframe for rows at (level, parent)
        mask = (
            (hierarchy_prob_df["hierarchy_level"] == level) &
            (hierarchy_prob_df["parent_node"] == parent)
        )
        siblings_df = hierarchy_prob_df[mask]
        C = len(siblings_df)
        if C <= 1:
            continue  # Only 1 or no child => no "siblings" to update

        # Identify the row for 'node'
        node_mask = mask & (hierarchy_prob_df["child_node"] == node)
        if not node_mask.any():
            # The node is not found under this parent, skip
            continue

        current_prob = hierarchy_prob_df.loc[node_mask, "probability"].values[0]

        # Define small increments for node vs. siblings
        delta_node = 1 / (10 * C)
        delta_sibling = 1 / (10 * C * (C - 1))

        # If cdf_val > 0.5 => increase node probability, decrease siblings
        if cdf_val > 0.5:
            new_prob_node = current_prob + delta_node
            hierarchy_prob_df.loc[node_mask, "probability"] = new_prob_node

            sibling_mask = mask & (hierarchy_prob_df["child_node"] != node)
            hierarchy_prob_df.loc[sibling_mask, "probability"] -= delta_sibling
        else:
            # Otherwise, decrease node probability, increase siblings
            new_prob_node = current_prob - delta_node
            hierarchy_prob_df.loc[node_mask, "probability"] = new_prob_node

            sibling_mask = mask & (hierarchy_prob_df["child_node"] != node)
            hierarchy_prob_df.loc[sibling_mask, "probability"] += delta_sibling

        # Clamp to [0,1]
        hierarchy_prob_df.loc[mask, "probability"] = hierarchy_prob_df.loc[mask, "probability"].clip(0.0, 1.0)

        # Re-normalize so children sum to 1
        new_probs = hierarchy_prob_df.loc[mask, "probability"].values
        sum_probs = new_probs.sum()
        if sum_probs > 0:
            normalized = new_probs / sum_probs
            hierarchy_prob_df.loc[mask, "probability"] = normalized
        else:
            # If sum_probs=0, distribute uniformly
            uniform_val = 1.0 / C
            hierarchy_prob_df.loc[mask, "probability"] = uniform_val

    return hierarchy_prob_df

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Update Hierarchy Probabilities Based on Random Walks")
    parser.add_argument("--random_walk_csv", type=str, default="walks/random_walk_draws.csv",
                        help="Path to random_walk_draws.csv")
    parser.add_argument("--hierarchy_prob_csv", type=str, default="probs/hierarchy_probabilities.csv",
                        help="Path to hierarchy_probabilities.csv")
    parser.add_argument("--hierarchy_mapping_csv", type=str, default="maps/imagenet_hierarchy_mapping.csv",
                        help="Path to imagenet_hierarchy_mapping.csv")
    parser.add_argument("--distribution_csv", type=str, default="distribution/out_dist.csv",
                        help="Path to distribution CSV containing mean and std")
    parser.add_argument("--model_weights", type=str, default="models/resnet18_10classes.pth",
                        help="Path to the trained ResNet-18 model weights")
    parser.add_argument("--output_prob", type=str, default="probs/hierarchy_probabilities_new.csv",
                        help="Path to output file")
    parser.add_argument("--epoch", type=int, default=0,
                        help="Current epoch number, used for logging the mean CDF to results CSV")
    parser.add_argument("--results_csv", type=str, default="./results/result.csv",
                        help="File to append (epoch, meanCDF) records")
    os.makedirs(os.path.dirname('./results'), exist_ok=True)
    args = parser.parse_args()

    # Configuration
    RANDOM_WALK_CSV = args.random_walk_csv
    HIERARCHY_PROB_CSV = args.hierarchy_prob_csv
    MAPPING_CSV = args.hierarchy_mapping_csv
    DISTRIBUTION_CSV = args.distribution_csv
    MODEL_WEIGHTS = args.model_weights

    # The correct variable name for updated probabilities
    UPDATED_HIERARCHY_PROB_CSV = args.output_prob

    # 1) Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2) Load Distribution CSV
    if not os.path.exists(DISTRIBUTION_CSV):
        raise FileNotFoundError(f"Distribution CSV not found: {DISTRIBUTION_CSV}")
    dist_df = pd.read_csv(DISTRIBUTION_CSV)
    if not {"param", "value"}.issubset(dist_df.columns):
        raise ValueError("Distribution CSV must have 'param' and 'value' columns.")

    mean_val = dist_df.loc[dist_df["param"] == "mean", "value"].values
    std_val = dist_df.loc[dist_df["param"] == "std", "value"].values

    if len(mean_val) == 0 or len(std_val) == 0:
        raise ValueError("Distribution CSV must contain 'mean' and 'std' parameters.")

    mean_val, std_val = mean_val[0], std_val[0]
    normal_dist = norm(loc=mean_val, scale=std_val)
    print(f"Loaded normal distribution: mean={mean_val:.3f}, std={std_val:.3f}")

    # 3) Load Mapping CSV
    if not os.path.exists(MAPPING_CSV):
        raise FileNotFoundError(f"Hierarchy mapping CSV not found: {MAPPING_CSV}")
    mapping_df = pd.read_csv(MAPPING_CSV)
    hier_cols = sorted([c for c in mapping_df.columns if c.startswith("hierarchy_")],
                       key=lambda x: int(x.split("_")[1]))
    if not hier_cols:
        raise ValueError("No hierarchy columns found in imagenet_hierarchy_mapping.csv")
    print(f"Hierarchy mapping loaded with columns: {hier_cols}")

    # 4) Load Hierarchy Probabilities
    if not os.path.exists(HIERARCHY_PROB_CSV):
        raise FileNotFoundError(f"Hierarchy probabilities CSV not found: {HIERARCHY_PROB_CSV}")
    hierarchy_prob_df = pd.read_csv(HIERARCHY_PROB_CSV)
    req_cols = {"hierarchy_level", "parent_node", "child_node", "probability"}
    if not req_cols.issubset(hierarchy_prob_df.columns):
        raise ValueError(f"Missing columns in hierarchy_probabilities.csv: {req_cols - set(hierarchy_prob_df.columns)}")
    print("Hierarchy probabilities loaded.")

    # 5) Load Random Walk CSV
    if not os.path.exists(RANDOM_WALK_CSV):
        raise FileNotFoundError(f"Random walk CSV not found: {RANDOM_WALK_CSV}")
    walks_df = pd.read_csv(RANDOM_WALK_CSV)
    needed_cols = {"walk_id", "hierarchy_path", "chosen_wnid", "chosen_image"}
    if not needed_cols.issubset(walks_df.columns):
        raise ValueError(f"Missing columns in random_walk_draws.csv: {needed_cols - set(walks_df.columns)}")
    print("Random walk draws loaded.")

    cdf_values = []
    current_epoch = args.epoch
    results_csv = args.results_csv

    for idx, row in walks_df.iterrows():
        walk_id = row["walk_id"]
        hierarchy_path = row["hierarchy_path"]
        chosen_wnid = row["chosen_wnid"]
        chosen_image = row["chosen_image"]

        if pd.isna(hierarchy_path) or hierarchy_path.strip() == "":
            print(f"Walk ID {walk_id}: Empty hierarchy_path. Skipping.")
            continue

        if pd.isna(chosen_image) or chosen_image.strip() == "":
            print(f"Walk ID {walk_id}: No image selected. Skipping.")
            continue

        # Compute the logit gap's CDF
        gap, logit_val, cdf_val = compute_cdf_of_logit_gap(
            chosen_image, model, device, inference_transform, normal_dist
        )
        if gap is None:
            print(f"Walk ID {walk_id}: Failed to compute logit gap. Skipping.")
            continue

        # Log the result to console
        print(f"Walk ID {walk_id}: logit_gap={gap:.4f}, logit_transform={logit_val:.4f}, CDF={cdf_val:.4f}")

        # Save the cdf to cdf_values
        cdf_values.append(cdf_val)

        # Retrieve the path for the chosen wnid
        wnid_row = mapping_df[mapping_df["wnid"] == chosen_wnid]
        if wnid_row.empty:
            print(f"Walk ID {walk_id}: WNID '{chosen_wnid}' not found in mapping. Skipping.")
            continue

        path_list = wnid_row.iloc[0][hier_cols].dropna().tolist()

        # (6) Update probabilities
        hierarchy_prob_df = update_probabilities(path_list, cdf_val, hierarchy_prob_df, mapping_df, hier_cols)

    # (7) Save Updated Hierarchy Probabilities
    hierarchy_prob_df.to_csv(UPDATED_HIERARCHY_PROB_CSV, index=False)
    print(f"Updated hierarchy probabilities saved to {UPDATED_HIERARCHY_PROB_CSV}")

    # (8) Compute mean CDF for this run and append to results CSV
    if cdf_values:
        mean_cdf = sum(cdf_values) / len(cdf_values)
        print(f"Mean CDF for epoch {current_epoch} => {mean_cdf:.4f}")

        # Ensure parent folder exists
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)

        # Append a line: "epoch,meanCDF"
        with open(results_csv, "a") as f:
            f.write(f"{current_epoch},{mean_cdf:.6f}\n")
    else:
        print(f"No valid CDF values were computed in this run. Skipping CSV log entry.")

if __name__ == "__main__":
    main()

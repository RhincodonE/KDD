import os
import random
import argparse
import pandas as pd
import os

# ----------------------------------------------------------------------
# Helper Function: Perform a Sequential Random Walk
# ----------------------------------------------------------------------
def sample_sequential_random_path(hierarchy_distribution):
    """
    Performs a sequential random walk respecting the hierarchy.
    Starts at the root node and samples child nodes based on parent-child relationships.

    Args:
        hierarchy_distribution (dict): Mapping of hierarchy_level -> parent_node -> list of (child_node, probability)

    Returns:
        list: A list of node names representing the path.
    """
    path = []
    current_parent = "(none)"  # Starting with '(none)' as per hierarchy_probabilities.csv

    for level in sorted(hierarchy_distribution.keys()):
        if current_parent not in hierarchy_distribution[level]:
            # No children to sample from; terminate the path
            break

        children = hierarchy_distribution[level][current_parent]
        child_nodes, probabilities = zip(*children)  # Separate names and probs

        chosen_child = random.choices(child_nodes, weights=probabilities, k=1)[0]
        path.append(chosen_child)
        current_parent = chosen_child  # Update the current parent for the next level

    return path

# ----------------------------------------------------------------------
# Helper Function: Find Matching WNIDs
# ----------------------------------------------------------------------
def find_matching_wnids(path, map_df, hier_cols):
    """
    Finds all WNIDs in the hierarchy mapping that exactly match the given path.

    Args:
        path (list): A list of node names representing the path.
        map_df (pd.DataFrame): DataFrame containing WNIDs and their hierarchical paths.
        hier_cols (list): Sorted list of hierarchy column names.

    Returns:
        pd.DataFrame: A subset of map_df containing matching WNIDs.
    """
    df_filtered = map_df.copy()
    for i, node in enumerate(path):
        if i >= len(hier_cols):
            # Path extends beyond available hierarchy levels
            return pd.DataFrame()
        col = hier_cols[i]
        df_filtered = df_filtered[df_filtered[col] == node]
        if df_filtered.empty:
            break
    return df_filtered

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------
def main():
    # ----------------------------------------------------------------------
    # Argument Parsing
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Perform Sequential Random Walks on ImageNet Hierarchy")
    parser.add_argument(
        "--custom_hierarchy_csv",
        type=str,
        default="maps/imagenet_hierarchy_mapping.csv",
        help="Path to imagenet_hierarchy_mapping.csv"
    )
    parser.add_argument(
        "--hierarchy_prob_csv",
        type=str,
        default="probs/hierarchy_probabilities.csv",
        help="Path to hierarchy_probabilities.csv"
    )
    os.makedirs(os.path.dirname('./probs'), exist_ok=True)
    parser.add_argument(
        "--num_walks",
        type=int,
        default=50,
        help="Number of random walks to perform (K)"
    )
    parser.add_argument(
        "--output_results_csv",
        type=str,
        default="walks/random_walk_draws.csv",
        help="Path to save the random walk results CSV"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=os.path.expanduser("~/kekechen_common/datasets/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train"),
        help="Path to the ImageNet training data directory"
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------
    CUSTOM_HIERARCHY_CSV = args.custom_hierarchy_csv
    HIERARCHY_PROB_CSV = args.hierarchy_prob_csv
    K = args.num_walks
    OUTPUT_RESULTS_CSV = args.output_results_csv
    TRAIN_DIR = args.train_dir

    # ----------------------------------------------------------------------
    # Step 1: Load Hierarchy Probabilities
    # ----------------------------------------------------------------------
    # We expect columns: [hierarchy_level, parent_node, child_node, probability]
    if not os.path.exists(HIERARCHY_PROB_CSV):
        raise FileNotFoundError(f"Hierarchy probabilities file not found: {HIERARCHY_PROB_CSV}")

    hier_prob_df = pd.read_csv(HIERARCHY_PROB_CSV)

    # Validate required columns
    required_columns = {"hierarchy_level", "parent_node", "child_node", "probability"}
    if not required_columns.issubset(set(hier_prob_df.columns)):
        missing = required_columns - set(hier_prob_df.columns)
        raise ValueError(f"Missing columns in hierarchy_probabilities.csv: {missing}")

    # Group by 'hierarchy_level' to structure the probability distributions
    grouped = hier_prob_df.groupby("hierarchy_level")

    # Create a mapping: hierarchy_level -> parent_node -> list of (child_node, probability)
    hierarchy_distribution = {}
    for level, group in grouped:
        hierarchy_distribution[level] = {}
        for _, row in group.iterrows():
            parent = row["parent_node"]
            child = row["child_node"]
            prob = row["probability"]
            if parent not in hierarchy_distribution[level]:
                hierarchy_distribution[level][parent] = []
            hierarchy_distribution[level][parent].append((child, prob))

    print("Hierarchy distribution loaded and structured.")

    # ----------------------------------------------------------------------
    # Step 2: Load the Hierarchy Mapping (WNIDs + Hierarchy Columns)
    # ----------------------------------------------------------------------
    if not os.path.exists(CUSTOM_HIERARCHY_CSV):
        raise FileNotFoundError(f"Custom hierarchy mapping file not found: {CUSTOM_HIERARCHY_CSV}")

    map_df = pd.read_csv(CUSTOM_HIERARCHY_CSV)

    # Identify which columns are hierarchy columns
    hier_cols = [c for c in map_df.columns if c.startswith("hierarchy_")]
    hier_cols = sorted(hier_cols, key=lambda x: int(x.split("_")[1]))  # Ensure sorted by level

    if not hier_cols:
        raise ValueError("No hierarchy columns found in imagenet_hierarchy_mapping.csv")

    print(f"Hierarchy mapping loaded with columns: {hier_cols}")

    # ----------------------------------------------------------------------
    # Step 3: Perform K Sequential Random Walks and Select Images
    # ----------------------------------------------------------------------
    results = []
    for walk_id in range(1, K + 1):
        # Perform a sequential random path
        random_path = sample_sequential_random_path(hierarchy_distribution)

        # Convert path list to string
        path_str = "-".join(random_path)

        # Find matching WNIDs
        candidates_df = find_matching_wnids(random_path, map_df, hier_cols)

        if candidates_df.empty:
            # No matching WNIDs found for this path
            chosen_wnid = None
            chosen_image = None
            print(f"Walk ID {walk_id}: No matching WNID found for path '{path_str}'")
        else:
            # Randomly select one WNID from the candidates
            chosen_row = candidates_df.sample(n=1).iloc[0]
            chosen_wnid = chosen_row["wnid"]

            # Randomly select one image from the chosen WNID's folder
            wnid_folder = os.path.join(TRAIN_DIR, chosen_wnid)
            if not os.path.isdir(wnid_folder):
                chosen_image = None
                print(f"Walk ID {walk_id}: WNID folder '{chosen_wnid}' does not exist.")
            else:
                images = [img for img in os.listdir(wnid_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not images:
                    chosen_image = None
                    print(f"Walk ID {walk_id}: No images found in folder '{chosen_wnid}'.")
                else:
                    chosen_img_file = random.choice(images)
                    chosen_image = os.path.abspath(os.path.join(wnid_folder, chosen_img_file))
                    print(f"Walk ID {walk_id}: Chose image '{chosen_image}' from WNID '{chosen_wnid}'.")

        # Store the results
        results.append({
            "walk_id": walk_id,
            "hierarchy_path": path_str,
            "chosen_wnid": chosen_wnid,
            "chosen_image": chosen_image
        })

    # ----------------------------------------------------------------------
    # Step 4: Save the Results to CSV
    # ----------------------------------------------------------------------
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"Random walk draws saved to {OUTPUT_RESULTS_CSV}")

if __name__ == "__main__":
    main()

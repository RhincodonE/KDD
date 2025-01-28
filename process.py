import os
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import nltk
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_DIR = os.path.expanduser(
    "~/kekechen_common/datasets/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train"
)
OUTPUT_MAPPING_CSV = "maps/imagenet_hierarchy_mapping.csv"
os.makedirs(os.path.dirname('./maps'), exist_ok=True)

OUTPUT_PROB_CSV = "probs/hierarchy_probabilities.csv"
os.makedirs(os.path.dirname('./probs'), exist_ok=True)

nltk.data.path.append('~/kekechen_common/datasets/imagenet/imagenet/nltk_data')
nltk.download('wordnet', download_dir='~/kekechen_common/datasets/imagenet/imagenet/nltk_data')

# ----------------------------------------------------------------------
# Step 1: Collect all WNIDs from the train directory
# ----------------------------------------------------------------------
if not os.path.isdir(TRAIN_DIR):
    raise ValueError(f"Train directory not found: {TRAIN_DIR}")

# All subfolders in TRAIN_DIR are WNIDs like 'n01440764'
wnids = [
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d)) and d.startswith("n")
]
print(wnids)

print(f"Found {len(wnids)} WNIDs in: {TRAIN_DIR}")

# ----------------------------------------------------------------------
# Step 2: Build a DataFrame that maps each WNID to its WordNet hypernym path
# ----------------------------------------------------------------------
rows = []

for wnid in tqdm(wnids, desc="Processing WNIDs"):
    # WordNet offset: 'n01440764' => offset = 1440764
    # Strip the leading 'n'
    offset_str = wnid[1:]
    try:
        offset_int = int(offset_str)
    except ValueError:
        # If not a valid integer offset, skip
        continue

    # Create the synset (noun 'n')
    # This can raise an error if the offset is invalid or not in WordNet
    try:
        syn = wn.synset_from_pos_and_offset('n', offset_int)
    except Exception as e:
        print(f"Offset {offset_int} (WNID: {wnid}) not found in this WordNet. Error: {e}")
        continue

    # Get the first hypernym path
    # e.g. [Synset('entity.n.01'), ..., Synset('shark.n.01'), Synset('white_shark.n.01')]
    paths = syn.hypernym_paths()
    if not paths:
        continue
    hyper_path = paths[0]  # we only take the first path if multiple

    # Convert each Synset to a lemma name for a "hierarchy" entry
    # e.g. "white_shark"
    # Note that some synsets have multiple lemma names; we use the first one.
    path_lemmas = [s.lemmas()[0].name() for s in hyper_path]

    # We'll store as columns: [wnid, hierarchy_0, hierarchy_1, ...]
    row_dict = {"wnid": wnid}
    for i, lemma in enumerate(path_lemmas):
        row_dict[f"hierarchy_{i}"] = lemma

    rows.append(row_dict)

df = pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Step 3: Save the WNID -> Hierarchy mapping
# ----------------------------------------------------------------------
df.to_csv(OUTPUT_MAPPING_CSV, index=False)
print(f"Saved WordNet hierarchies for each WNID to {OUTPUT_MAPPING_CSV}")

# ----------------------------------------------------------------------
# Step 4: Compute Sibling-Based Probabilities
# ----------------------------------------------------------------------
# We'll interpret each 'hierarchy_i' as a level i. Then for each parent at level i,
# we find all children at level (i+1). Each child gets 1 / (number_of_children).
# We'll store the results in columns:
#   [hierarchy_level, parent_node, child_node, probability]

# Identify all hierarchy columns
hier_cols = [c for c in df.columns if c.startswith("hierarchy_")]
hier_cols = sorted(hier_cols, key=lambda x: int(x.split("_")[1]))  # ensure ascending order

# Build parent->children map for each level
parent_to_children = {}
max_level = len(hier_cols)

for level in range(max_level - 1):  # up to second-to-last
    parent_col = hier_cols[level]
    child_col  = hier_cols[level + 1]

    # For each row, parent = row[parent_col], child = row[child_col]
    # Collect into a dict of sets
    p2c_map = {}
    valid_rows = df.dropna(subset=[parent_col, child_col])
    for _, row_data in valid_rows.iterrows():
        parent = row_data[parent_col]
        child = row_data[child_col]
        if parent not in p2c_map:
            p2c_map[parent] = set()
        p2c_map[parent].add(child)

    parent_to_children[level] = p2c_map

# We'll also store the distinct nodes at each level (not strictly needed, but can be handy)
level_nodes = {}
for i, col in enumerate(hier_cols):
    level_nodes[i] = set(df[col].dropna().unique())

# Now compute probabilities
results_prob = []

# There's typically a top-level root "entity" at hierarchy_0,
# so let's mark it with probability = 1.0 (if you prefer).
root_nodes = level_nodes.get(0, [])
for root in root_nodes:
    results_prob.append({
        "hierarchy_level": 0,
        "parent_node": "(none)",
        "child_node": root,
        "probability": 1.0
    })

for level in range(max_level - 1):
    p2c_map = parent_to_children.get(level, {})
    for parent_node, children_set in p2c_map.items():
        children_list = sorted(children_set)
        n_children = len(children_list)
        if n_children == 0:
            continue
        prob = 1.0 / n_children
        for child_node in children_list:
            results_prob.append({
                "hierarchy_level": level + 1,
                "parent_node": parent_node,
                "child_node": child_node,
                "probability": prob
            })

prob_df = pd.DataFrame(results_prob)
prob_df = prob_df.sort_values(
    by=["hierarchy_level", "parent_node", "child_node"]
).reset_index(drop=True)

prob_df.to_csv(OUTPUT_PROB_CSV, index=False)
print(f"Sibling-based probabilities saved to {OUTPUT_PROB_CSV}.")

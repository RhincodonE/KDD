import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_CSV = "obs/logit_differences.csv"   # Must contain a 'logit_difference' column
OUTPUT_DIST_CSV = "distribution/out_dist.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_DIST_CSV), exist_ok=True)

# --------------------------------------------------
# Step 1: Read CSV
# --------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# --------------------------------------------------
# Step 2: Compute log(x / (1 - x)) for valid x in (0,1)
# --------------------------------------------------
logit_transformed_values = []
for x in df["logit_difference"]:
    if 0 < x < 1:
        val = np.log(x / (1.0 - x))
        logit_transformed_values.append(val)

logit_transformed_values = np.array(logit_transformed_values)

# --------------------------------------------------
# Step 3: Fit a Normal Distribution (mean, std)
# --------------------------------------------------
if len(logit_transformed_values) == 0:
    print("No valid values in the range (0,1); cannot compute mean/std.")
    mean_val = float('nan')
    std_val = float('nan')
else:
    mean_val = logit_transformed_values.mean()
    std_val = logit_transformed_values.std(ddof=0)  # population std if you want sample-based use ddof=1

# --------------------------------------------------
# Step 4: Print Results and Save to CSV
# --------------------------------------------------
print(f"Number of valid samples: {len(logit_transformed_values)}")
print(f"Mean of log(x/(1-x)):   {mean_val:.6f}")
print(f"Std of log(x/(1-x)):    {std_val:.6f}")

# Save parameters to CSV
dist_df = pd.DataFrame({
    "param": ["mean", "std"],
    "value": [mean_val, std_val]
})
dist_df.to_csv(OUTPUT_DIST_CSV, index=False)
print(f"Saved distribution parameters to {OUTPUT_DIST_CSV}")

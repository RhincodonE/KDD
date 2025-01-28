
# LDI: Brand-New Implementation

This repository provides a fresh implementation of **LDI** (LiRA-based domain inference attack). The workflow consists of several scripts to **initialize hierarchies**, **train models**, **generate observations**, **produce distributions**, and **run attacks**.

## Overview

1. **`process.py`**  
   Initializes both the hierarchy mapping and the hierarchy probability file. These files are used later to perform attacks.

2. **`train_model.py`**  
   Trains one or more models (e.g., a “shadow model”) used for the attack or defense steps.

3. **`generate_obs.py`**  
   Generates observations on **non-member** data points. This is typically used to help produce distribution estimates.

4. **`distribution_generation.py`**  
   Builds the non-member *Gaussian distribution* based on your observations.

## Running an Attack

1. **Adjust `attack.sh`**  
   - In the first steps, you point to the outputs you created from **`process.py`**, **`train_model.py`**, **`generate_obs.py`**, and **`distribution_generation.py`**.  
   - Make sure to specify the number of **attack epochs** you want to perform.

2. **`walk_hierarchy.py` & `update_probs.py`**  
   - **Per epoch**:
     - `walk_hierarchy.py` performs a random walk over your hierarchy using the current probabilities.
     - `update_probs.py` then **updates** those hierarchy probabilities based on the chosen target model and the random walk results.

3. **Inspect Attack Results**  
   - After multiple epochs, you will have updated hierarchy probability files in `./probs/`.  
   - Select the final epoch (or any intermediate state) as your **attack output**.

## File Structure

- **`process.py`** – Creates hierarchy mapping (`imagenet_hierarchy_mapping.csv`) and hierarchy probability (`hierarchy_probabilities.csv`).  
- **`train_model.py`** – Trains a model (or shadow model).  
- **`generate_obs.py`** – Generates observation data from non-member samples.  
- **`distribution_generation.py`** – Builds the non-member distribution.  
- **`walk_hierarchy.py`** – Performs a random walk each epoch.  
- **`update_probs.py`** – Updates hierarchy probabilities based on the random walk and target model.  
- **`attack.sh`** – Example script to orchestrate the entire attack pipeline (run multiple epochs, random walks, probability updates, etc.).

## Quickstart

1. **Initialize**:
   ```bash
   python process.py
   ```

2. **Train Models**:
   ```bash
   python train_model.py
   ```

3. **Generate Observations & Distributions**:
   ```bash
   python generate_obs.py
   python distribution_generation.py
   ```

4. **Attack**:
   - **Edit `attack.sh`** to point to the files generated in steps 1–3.
   - **Run**:
     ```bash
     bash attack.sh
     ```
   - Find updated hierarchy probability files inside `./probs/`.


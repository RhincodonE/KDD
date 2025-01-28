#!/usr/bin/env bash



target_model_addr="models/resnet18_10classes.pth"
epochs=10
batch=100

for ((i=0; i<epochs; i++)); do

  # Current random walk output
  current_walk="walks/random_walk_$i.csv"

  # If i==0, use the base "hierarchy_probabilities.csv", else use the file from previous iteration
  if [ "$i" -eq 0 ]; then
    current_probs="probs/hierarchy_probabilities.csv"
  else
    prev=$((i - 1))
    current_probs="probs/hierarchy_probabilities_${prev}.csv"
  fi

  # Updated file for this iteration
  updated_probs="probs/hierarchy_probabilities_${i}.csv"

  # 1) Generate random walks based on the current probabilities
  python walk_hierarchy.py \
    --num_walks "$batch" \
    --output_results_csv "$current_walk" \
    --hierarchy_prob_csv "$current_probs"

  # 2) Update probabilities using the random walks + model
  python update_probs.py \
    --random_walk_csv "$current_walk" \
    --hierarchy_prob_csv "$current_probs" \
    --model_weights "$target_model_addr" \
    --output_prob "$updated_probs"  \
    --epoch $i


done

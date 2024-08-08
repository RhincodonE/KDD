import os
import pandas as pd
import numpy as np

def calculate_mean_std(folder_path):
    # Create a dictionary to store data for each index
    data = {}

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            for index, row in df.iterrows():
                idx = row['index']
                prediction = row['Prediction']

                if idx not in data:
                    data[idx] = []
                data[idx].append(prediction)

    # Calculate mean and standard deviation for each index
    results = {'index': [], 'mean_prediction': [], 'std_prediction': []}

    for idx, predictions in data.items():
        results['index'].append(idx)
        results['mean_prediction'].append(np.mean(predictions))
        results['std_prediction'].append(np.std(predictions))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

# Set the folder path containing the CSV files
folder_path = '/users/home/ygu27/cifar/MIA/data/results/out/'

# Calculate mean and std
results_df = calculate_mean_std(folder_path)

# Save the results to a new CSV file
results_df.to_csv('/users/home/ygu27/cifar/MIA/data/mean_std_predictions.csv', index=False)

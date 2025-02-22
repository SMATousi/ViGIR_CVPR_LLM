import os
import json
import re

# Define the directory containing the JSON files
directory = "./"  # Change this to your actual directory path

# Dictionary to store train and test files for each model
model_files = {}

# Regular expression to match train and test JSON files
pattern = re.compile(r"(.+)-(.+)-(train|test)\.json")

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json") and not filename.endswith("raw_info.json"):
        match = pattern.match(filename)
        if match:
            dataset_name, model_name, split = match.groups()
            key = f"{dataset_name}_{model_name}"
            
            if key not in model_files:
                model_files[key] = {"train": None, "test": None}
            
            model_files[key][split] = filename

# Merge the train and test files for each model
for key, files in model_files.items():
    train_file = files["train"]
    test_file = files["test"]
    
    merged_data = []

    if train_file:
        with open(os.path.join(directory, train_file), "r") as f:
            merged_data.extend(json.load(f))

    if test_file:
        with open(os.path.join(directory, test_file), "r") as f:
            merged_data.extend(json.load(f))
    
    # Write the merged data to a new file
    output_file = os.path.join(directory, f"{key}.json")
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged files into {output_file}")

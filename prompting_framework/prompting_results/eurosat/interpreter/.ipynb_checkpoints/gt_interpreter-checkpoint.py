import json
import os

# Load JSON file
with open('../eurosat-llava-phi3-train-raw_info.json', 'r') as file:
    data = json.load(file)

# Dictionary to store the results with image name and label
results = {}

# Process each entry in the JSON file
for image_path, info in data.items():
    # Extract the image name from the full path
    image_name = os.path.basename(image_path)
    
    # Get the label from the info dictionary
    label = info.get("label", None)
    
    # Store the image name and label
    results[image_name] = label

# Save results to a new JSON file
with open('train_gt.json', 'w') as result_file:
    json.dump(results, result_file, indent=4)

print("Results saved to 'image_name_label_results.json'")

import json
import re
import os

list_of_all_files = os.listdir('../')
print(list_of_all_files)
# List of target classes and their assigned numbers
class_numbers = {
    "SeaLake": 9,
    "PermanentCrop": 6,
    "River": 8,
    "Residential": 7,
    "Pasture": 5,
    "Industrial": 4,
    "Highway": 3,
    "HerbaceousVegetation": 2,
    "Forest": 1,
    "AnnualCrop": 0
}

for file_name in list_of_all_files:

    if file_name.endswith('info.json'):

        continue
    
    elif file_name.endswith('test.json'):

        with open(os.path.join('../', file_name), 'r') as file:
            data = json.load(file)

        
        # Dictionary to store the results
        results = {}

        # Process each image entry in the JSON file
        for image_path, response in data.items():
            # Extract the image name from the full path
            image_name = os.path.basename(image_path)
            
            detected_classes = []

            # Check each target class and see if it appears in the response text
            for class_name, class_number in class_numbers.items():
                # Case insensitive regex for finding the class
                if re.search(class_name, response, re.IGNORECASE):
                    detected_classes.append(class_number)

            # Store results based on the number of detected classes
            if len(detected_classes) > 1:
                results[image_name] = -1  # More than one class found
            elif len(detected_classes) == 1:
                results[image_name] = detected_classes[0]  # Single class found

        # Save results to a new JSON file
        with open(file_name+'_results.json', 'w') as result_file:
            json.dump(results, result_file, indent=4)

        print("Results saved to 'results.json'")

import json
import re
import os

list_of_all_files = os.listdir('../')
print(list_of_all_files)
# List of target classes and their assigned numbers
class_numbers = {
    "abyssinian": 0,
    "american_bulldog": 1,
    "american_pit_bull_terrier": 2,
    "basset_hound": 3,
    "beagle": 4,
    "bengal": 5,
    "birman": 6,
    "bombay": 7,
    "boxer": 8,
    "british_shorthair": 9,
    "chihuahua": 10,
    "egyptian_mau": 11,
    "english_cocker_spaniel": 12,
    "english_setter": 13,
    "german_shorthaired": 14,
    "great_pyrenees": 15,
    "havanese": 16,
    "japanese_chin": 17,
    "keeshond": 18,
    "leonberger": 19,
    "maine_coon": 20,
    "miniature_pinscher": 21,
    "newfoundland": 22,
    "persian": 23,
    "pomeranian": 24,
    "pug": 25,
    "ragdoll": 26,
    "russian_blue": 27,
    "saint_bernard": 28,
    "samoyed": 29,
    "scottish_terrier": 30,
    "shiba_inu": 31,
    "siamese": 32,
    "sphynx": 33,
    "staffordshire_bull_terrier": 34,
    "wheaten_terrier": 35,
    "yorkshire_terrier": 36
}

for file_name in list_of_all_files:

    if file_name.endswith('info.json'):

        continue
    
    elif file_name.endswith('train.json'):

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

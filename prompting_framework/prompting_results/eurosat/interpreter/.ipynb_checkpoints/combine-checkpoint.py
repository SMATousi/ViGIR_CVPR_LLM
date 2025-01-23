import json
import sys

def combine_json_files(file1, file2, output_file):
    # Load the first JSON file
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    # Load the second JSON file
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # Combine the dictionaries
    combined_data = {**data1, **data2}
    
    # Save combined data to the specified output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined JSON saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_json.py <file1.json> <file2.json> <output_file.json>")
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        output_file = sys.argv[3]
        combine_json_files(file1, file2, output_file)

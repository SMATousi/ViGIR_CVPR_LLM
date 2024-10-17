import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Convert JSON file format")
parser.add_argument("input_file", type=str, help="Path to the input JSON file")
parser.add_argument("output_file", type=str, help="Path to the output JSON file")

# Parse arguments
args = parser.parse_args()

# Load the original JSON file
with open(args.input_file, "r") as f:
    data = json.load(f)

# Create the new format
new_format = [{"img": key, "label": value} for key, value in data.items()]

# Save the new format to a file
with open(args.output_file, "w") as f:
    json.dump(new_format, f, indent=4)

print(f"Converted JSON has been saved to {args.output_file}")


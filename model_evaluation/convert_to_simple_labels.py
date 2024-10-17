import json

#Input and output file paths
input_file = 'dev.jsonl'
output_file = 'simplified_dev.json'

simplified_data = []


with open(input_file, 'r') as file:
    for line in file:
        entry = json.loads(line)
        simplified_entry = {
            "img": f'{entry["id"]}.png',
            "label": entry["label"]
            }

        simplified_data.append(simplified_entry)


with open(output_file, 'w') as outfile:
    json.dump(simplified_data, outfile, indent=4)

print(f'Done {output_file}')



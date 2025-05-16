import json

# Path to the JSON file
input_path = r"SciVQA\SpiQa\SPIQA_train.json"
output_path = r"SciVQA\SpiQa\SPIQA_train_fixed.json"

# Read and reformat the JSON file
with open(input_path, 'r', encoding='utf-8', errors='replace') as file:
	data = json.load(file)

# Write the properly indented JSON to a new file
with open(output_path, 'w', encoding='utf-8', errors='replace') as file:
	json.dump(data, file, indent=4)

print(f"Fixed JSON saved to {output_path}")

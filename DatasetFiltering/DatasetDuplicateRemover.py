import json
"""
Usage:
    The script loads two datasets, finds matching entries based on paper IDs, removes duplicates, and saves both the duplicates and the filtered dataset to disk.

Functions:
    preprocess_paper_id(paper_id):
        Preprocesses a paper ID by removing all non-numeric characters.

    longest_common_substring(s1, s2):
        Computes the length of the longest common substring between two strings.

    find_matching_instances(first_dataset, second_dataset):
        Finds entries in the first dataset whose extracted image IDs have a sufficiently long common substring with any paper ID in the second dataset.

    remove_and_save_duplicates(first_dataset, matches, output_file):
        Removes entries from the first dataset that match the given IDs, saves them as duplicates to a file, and returns the remaining entries.
"""
import re

# Load the first dataset (JSONL format)
def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# Load the second dataset (JSON format)
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_id_from_image(image_path):
    # Assumes image path like "images/0904.0709_0.jpg"
    filename = image_path.split('/')[-1].split('_')[0]
    return re.sub(r'\D', '', filename)  # Remove all non-numeric characters

def preprocess_paper_id(paper_id):
    return re.sub(r'\D', '', paper_id)  # Remove all non-numeric characters

def longest_common_substring(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    lcs_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                lcs_length = max(lcs_length, dp[i][j])

    return lcs_length

def find_matching_instances(first_dataset, second_dataset):
    paper_ids = {preprocess_paper_id(entry['paper_id']) for entry in second_dataset}

    matching_instances = []
    for i, entry in enumerate(first_dataset):
        if i % 1000 == 0:
            print(f"Processing entry {i} of {len(first_dataset)}")
        id_from_image = extract_id_from_image(entry['image'])
        for paper_id in paper_ids:
            if longest_common_substring(id_from_image, paper_id) > 6:
                matching_instances.append(id_from_image)
                break

    return matching_instances

def remove_and_save_duplicates(first_dataset, matches, output_file):
    duplicates = [entry for entry in first_dataset if extract_id_from_image(entry['image']) in matches]
    remaining_entries = [entry for entry in first_dataset if extract_id_from_image(entry['image']) not in matches]

    # Save duplicates to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, ensure_ascii=False, indent=2)

    return remaining_entries


# Replace with actual file paths
first_dataset_path = 'SciVQA/unsloth/arxivqa/arxivqa.jsonl'
second_dataset_path = 'SciVQA/unsloth/shared_task/wholeDataset.json'

first_dataset = load_jsonl(first_dataset_path)
second_dataset = load_json(second_dataset_path)

matches = find_matching_instances(first_dataset, second_dataset)

output_file_path = 'duplicate_entries.json'
remaining_entries = remove_and_save_duplicates(first_dataset, matches, output_file_path)
filtered_dataset_path = 'SciVQA/unsloth/arxivqa/filteredDataset.json'
with open(filtered_dataset_path, 'w', encoding='utf-8') as f:
    json.dump(remaining_entries, f, ensure_ascii=False, indent=2)

for match in matches:
    print(json.dumps(match, indent=2))

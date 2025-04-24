import re

# def detect_weird_unicode(file_path):
#     with open(file_path, encoding='utf-8') as f:
#         lines = f.readlines()

#     for i, line in enumerate(lines):
#         # Only allow basic printable ASCII + newline
#         weird_chars = re.findall(r'[^\x00-\x7F]', line)
#         if weird_chars:
#             print(f"Line {i+1}: {line.strip()}")
#             print(f"  Problematic characters: {set(weird_chars)}")

# # Example usage:
# detect_weird_unicode("WizardDataset.jsonl")





import json

input_filename = "WizardDataset.jsonl"
output_filename = "WizardDatasetCleaned.jsonl"

# Mapping of problematic unicode characters to standard ASCII
REPLACEMENTS = {
    '‘': "'",  # Left single quote
    '’': "'",  # Right single quote or apostrophe
    '“': '"',  # Left double quote
    '”': '"',  # Right double quote
    '—': '-',  # Em dash
    '–': '-',  # En dash
}

def clean_text(text):
    for bad_char, replacement in REPLACEMENTS.items():
        text = text.replace(bad_char, replacement)
    return text

with open(input_filename, "r", encoding="utf-8") as infile, \
     open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue
        data = json.loads(line)
        data['instruction'] = clean_text(data['instruction'])
        data['response'] = clean_text(data['response'])
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print("Cleaning complete. Output written to", output_filename)







# THIS FINDS BAD LINES VVVVVVVVVVVVVV
# import json

# input_filename = "WizardDataset.jsonl"
# output_filename = "WizardDatasetCleaned.jsonl"

# REPLACEMENTS = {
#     '‘': "'",
#     '’': "'",
#     '“': '"',
#     '”': '"',
#     '—': '-',
#     '–': '-',
# }

# def clean_text(text):
#     for bad_char, replacement in REPLACEMENTS.items():
#         text = text.replace(bad_char, replacement)
#     return text

# with open(input_filename, "r", encoding="utf-8") as infile, \
#      open(output_filename, "w", encoding="utf-8") as outfile:

#     for i, line in enumerate(infile, 1):
#         if not line.strip():
#             continue
#         try:
#             data = json.loads(line)
#             data["instruction"] = clean_text(data["instruction"])
#             data["response"] = clean_text(data["response"])
#             json.dump(data, outfile, ensure_ascii=False)
#             outfile.write("\n")
#         except json.JSONDecodeError as e:
#             print(f"Error parsing line {i}: {e}")
#             print(f"Line content: {line.strip()}")
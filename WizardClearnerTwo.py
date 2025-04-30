# this one is actually still relevant if i update the dataset. 
import json

input_filename = "WizardDataset.jsonl"   
output_filename = "WizardDatasetFormatted.jsonl" 
with open(input_filename, "r", encoding="utf-8") as infile, \
     open(output_filename, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        if not line.strip():
            continue
        data = json.loads(line)
        prompt = f"{data['instruction']}\n\n### Response:\n"
        completion = f"{data['response'].strip()}"
        formatted = {
            "prompt": prompt,
            "completion": completion
        }
        json.dump(formatted, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Reformatting complete. Output saved to {output_filename}")

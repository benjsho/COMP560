import json

input_path  = "dialogue.tsv"
output_path = "dialogue_dataset.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line or "\t" not in line:
            continue
        prompt, reply = line.split("\t", 1)
        record = {
            "instruction": prompt.strip(),
            "response":    reply.strip()
        }
        json.dump(record, fout, ensure_ascii=False)
        fout.write("\n")

print(f"i think i wrote JSONL to {output_path}")
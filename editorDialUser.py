import json

in_path  = "dialogue_dataset.jsonl"
out_path = "dialogue_with_tags.jsonl"

with open(in_path,  "r", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:

    for line in fin:
        rec = json.loads(line)
        user = rec["instruction"].strip()
        bot  = rec["response"].strip()
        if bot and bot[0].islower():
            bot = bot[0].upper() + bot[1:]

        text = f"User: {user}\nBot: {bot}"

        fout.write(json.dumps({ "text": text }) + "\n")

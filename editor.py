import pandas as pd
import re

# Load your raw dataset (make sure the filename matches)
df = pd.read_json('EBnpc_dataset.json', lines=True)

# Function to clean and reformat each example
def reformat_row(row):
    # Extract context description inside the brackets
    ctx = row['context']
    match = re.search(r'\[Context:\s*Player\s*([^]]+)\]', ctx)
    if match:
        instruction = f"Player {match.group(1).strip()}."
    else:
        instruction = ctx

    # Clean response: strip, normalize spaces, fix punctuation
    resp = row['response'].strip()
    resp = re.sub(r'\s+', ' ', resp)
    resp = re.sub(r'\s+([?!.,])', r'\1', resp)
    # Capitalize first letter if not already
    if resp:
        resp = resp[0].upper() + resp[1:]

    return pd.Series({
        'instruction': instruction,
        'response': resp
    })

# Apply transformation
cleaned = df.apply(reformat_row, axis=1)

print(cleaned.head(10).to_string(index=False))

# Save reformatted dataset  
cleaned.to_json('EBnpc_dataset_reformatted.json', orient='records', lines=True)

print("i think its done")
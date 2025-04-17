import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.getLogger("transformers").setLevel(logging.ERROR) 


print("dataset...")
dataset = load_dataset("json", data_files="EBnpc_dataset.json", split="train")
print("dataset example: \n")
print(dataset[0])

print("\n model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
print("\n model and tokenizer ready")

print("\n generation test")
input_text = "player interacts with Mr. Saturn"
input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

output_ids = model.generate(
    input_ids, 
    max_length=1000, 
    pad_token_id=tokenizer.eos_token_id, 
    temperature=0.7, #high means more random and diverse
    do_sample=True, 
    top_p=0.5 #high focuses on diversity, wider range of words 
)
#response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\"ai\": ", response)
print("Padding side:", tokenizer.padding_side)
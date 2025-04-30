# first program created for this project. 
# all of these programs are outdated because of the collab.
# they really only exist at this point to look back on. 
# the only thing in here not necessarily outdated is WizardDatasetFormatted(and its cleaner, WizardCleanerTwo.py).
# it's the 8,463 line dataset or whatever. 
# it's pretty damn good. i'd like to say i hand typed everything. 
# i didn't. 
# but i evaluated everything. 
# and hit ctrl c ctrl v hundreds of times in the process
# that counts for something, right? 
# 
#
#
# honestly, it shouldn't
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.getLogger("transformers").setLevel(logging.ERROR) 


print("dataset...")
dataset = load_dataset("json", data_files="dialogue_dataset.jsonl", split="train")
print("dataset example: \n")
print(dataset[0])
print("Type of example:", type(dataset[0]))

print("\n model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_AGAIN")
tokenizer.pad_token = tokenizer.eos_token
#model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model =  AutoModelForCausalLM.from_pretrained("./fine_tuned_AGAIN")
print("\n model and tokenizer ready")

print("\n generation test")
#input_text = "[Context] Player interacts with monkey\n[Response]\n"
input_text = "hey how are you?"
prompt_text = input_text + tokenizer.eos_token
input_ids   = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
#prompt = "[Context] Player interacts with man\n[Response]\n"
#input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

# output_ids = model.generate(
#     input_ids, 
#     max_length=40, 
#     pad_token_id=tokenizer.eos_token_id, 
#     temperature=0.7, #high means more random and diverse
#     do_sample=True, 
#     repetition_penalty=1.3,
#     no_repeat_ngram_size=2,
#     top_p=0.9, #high focuses on diversity, wider range of words 
#     eos_token_id=tokenizer.eos_token_id
# )

#for dialogue (online) dataset 
output_ids = model.generate(
    input_ids,
    max_new_tokens=12,
    do_sample=False,
    top_k=20,
    top_p=0.9,
    temperature=2.0,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,
)
# prompt_len   = input_ids.shape[-1]
prompt_len     = input_ids.size(-1)
new_token_ids  = output_ids[0, prompt_len:]  # everything after your prompt

reply = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

#response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
#response = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("[Response]")[-1].strip() # FOR EB

#response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip() # FOR DIALOGUE dataset (w/o dialgue dataset)


print("\"ai\":", reply)
print("Padding side:", tokenizer.padding_side)
# i completely forgot about this one. this one never came to fruition. 
# cool idea though. 
# it was when i was committed to the idea of training DialoGPT on an 
# already proven dataset, then fine tune that model again on some Earthbound s. 
# it didn't work. 
# i think fine tuning a four year old model was ultimately my main problem the whole time. 
# 
#
#
#
#
#
# who would've guessed. 

# finetune_dialoGPT on Cornell Movie-Dialogs Corpus using LoRA (PEFT)
# Simplified JSONL-based pipeline to fit resource constraints

# 1)Install prerequisites
#    pip install transformers datasets accelerate peft

# 2) Immmmports
import os
import json
import zipfile
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# 3) download & extract Cornell data (if not already)
DATA_DIR = Path("./cornell_corpus")
DATA_DIR.mkdir(exist_ok=True)
ZIP_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
ZIP_PATH = DATA_DIR / "cornell.zip"

if not ZIP_PATH.exists():
    import requests
    r = requests.get(ZIP_URL)
    ZIP_PATH.write_bytes(r.content)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(DATA_DIR)

# 4) parse raw files into JSONL-style examples
dialogs_path = DATA_DIR / "cornell movie-dialogs corpus" / "movie_conversations.txt"
lines_path   = DATA_DIR / "cornell movie-dialogs corpus" / "movie_lines.txt"

# Load line texts
id2line = {}
with open(lines_path, encoding='iso-8859-1') as f:
    for line in f:
        parts = line.split(" +++$+++ ")
        id2line[parts[0]] = parts[-1].strip()

# build pairs
pairs = []
with open(dialogs_path, encoding='iso-8859-1') as f:
    for conv in f:
        parts = conv.strip().split(" +++$+++ ")
        line_ids = eval(parts[-1])  # list of utterance IDs
        for i in range(len(line_ids) - 1):
            src = id2line[line_ids[i]]
            tgt = id2line[line_ids[i+1]]
            # build prompt-completion dict
            prompt = f"System: You are a professional, neutral assistant.\nUser: {src}\nBot:"
            completion = " " + tgt
            pairs.append({"text": prompt + completion})

# 5) create a Hugging Face dataset from list of dicts
dataset = Dataset.from_list(pairs)
# split into train/val (90/10)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
val_dataset   = dataset['test']

# 6) initialize tokenizer and prepare examples
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-medium", padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, max_length=512, padding=True
    )

train_tok = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
val_tok   = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

# 7) data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 8) Load pretrained DialoGPT and prepare LoRA
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium", load_in_8bit=True, device_map="auto"
)
model = prepare_model_for_int8_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["c_attn", "q_attn", "v_attn"],
    lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

# 9) TrainingArguments
training_args = TrainingArguments(
    output_dir="./dgpt_lora_cornell",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-4,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    fp16=True,
)

# 10) Initialize Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("./lora_adapter_cornell")

# 11) Optional chat loop (same as before)
def chat_loop():
    conv = "System: You are a professional, neutral assistant.\n"
    model.eval()
    while True:
        user = input(">> User: ")
        conv += f"User: {user}\nBot:"
        inputs = tokenizer(conv, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False, num_beams=2)
        reply = tokenizer.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        print("Bot:", reply)
        conv += f" {reply}\n"

# To start chatting, uncomment: chat_loop()


# # finetune_dialoGPT on DailyDialog using LoRA (PEFT) for resource-constrained setups
# # Combined end-to-end script with comments

# # 1) Install prerequisites
# #    pip install transformers datasets accelerate peft

# # 2) Imports
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM,
#     DataCollatorForLanguageModeling,
#     TrainingArguments, Trainer
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# # 3) Load DailyDialog dataset with its custom code
# #    Requires trust_remote_code to run the dataset script
# daily = load_dataset("daily_dialog", trust_remote_code=True)

# # 4) Preprocess: batched map to emit one example per adjacent utterance
# SYSTEM_PROMPT = "System: You are a professional, neutral assistant."

# def make_examples_batch(batch):
#     texts = []
#     for dialog in batch["dialog"]:
#         for i in range(len(dialog) - 1):
#             prompt = f"{SYSTEM_PROMPT}\nUser: {dialog[i]}\nBot:"
#             # leading space aids tokenization
#             texts.append(prompt + " " + dialog[i+1])
#     return {"text": texts}

# daily_columns = daily["train"].column_names
# train_dataset = daily["train"].map(
#     make_examples_batch, batched=True, remove_columns=daily_columns
# )
# val_dataset = daily["validation"].map(
#     make_examples_batch, batched=True, remove_columns=daily_columns
# )

# # 5) Initialize tokenizer and add pad token
# tokenizer = AutoTokenizer.from_pretrained(
#     "microsoft/DialoGPT-medium", padding_side="left"
# )
# tokenizer.pad_token = tokenizer.eos_token

# def tokenize(batch):
#     return tokenizer(
#         batch["text"], truncation=True, max_length=512, padding=True
#     )

# train_tok = train_dataset.map(
#     tokenize, batched=True, remove_columns=["text"]
# )
# val_tok = val_dataset.map(
#     tokenize, batched=True, remove_columns=["text"]
# )

# # 6) Data collator for causal LM
# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=False  # causal LM
# )

# # 7) Load pretrained DialoGPT and prepare for LoRA
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/DialoGPT-medium",
#     load_in_8bit=True,  # enable 8-bit quantization to save memory
#     device_map="auto",
# )
# model = prepare_model_for_int8_training(model)
# # LoRA configuration
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["c_attn", "q_attn", "v_attn"],
#     lora_dropout=0.05,
#     bias="none",
# )
# model = get_peft_model(model, lora_config)

# # 8) TrainingArguments
# from transformers import TrainingArguments
# training_args = TrainingArguments(
#     output_dir="./dgpt_lora_dailydialog",
#     per_device_train_batch_size=1,  # small batch for limited GPU
#     gradient_accumulation_steps=4,
#     num_train_epochs=1,
#     learning_rate=5e-4,
#     logging_steps=100,
#     evaluation_strategy="steps",
#     eval_steps=500,
#     save_total_limit=2,
#     fp16=True,
# )

# # 9) Initialize Trainer
# from transformers import Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_tok,
#     eval_dataset=val_tok,
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# # 10) Train with LoRA
# trainer.train()
# # Save only LoRA adapters
# model.save_pretrained("./lora_adapter")

# # 11) Optional inference

# def chat_loop():
#     conv = SYSTEM_PROMPT + "\n"
#     model.eval()
#     while True:
#         user = input(">> User: ")
#         conv += f"User: {user}\nBot:"
#         inputs = tokenizer(conv, return_tensors="pt").to(model.device)
#         out = model.generate(
#             **inputs,
#             max_new_tokens=60,
#             do_sample=False,
#             num_beams=2
#         )
#         reply = tokenizer.decode(
#             out[0, inputs.input_ids.shape[-1]:],
#             skip_special_tokens=True
#         ).strip()
#         print("Bot:", reply)
#         conv += f" {reply}\n"

# # To start chatting, uncomment: chat_loop()

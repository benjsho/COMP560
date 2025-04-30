# this is bad, dumb code VVVVVVVVVVVVV
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM,
#     Trainer, TrainingArguments
# )

# # 1) LOAD & SPLIT YOUR DATASET
# dataset = load_dataset(
#     "json",
#     data_files="dialogue_with_tags.jsonl",
#     split="train"
# ).train_test_split(test_size=0.1)

# # 2) LOAD MODEL & TOKENIZER
# model_name = "microsoft/DialoGPT-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
# tokenizer.pad_token = tokenizer.eos_token
# model     = AutoModelForCausalLM.from_pretrained(model_name)

# # 3) PREPROCESS: CONCAT CONTEXT + RESPONSE & CREATE LABELS
# def preprocess(example):
#     # Only input is the context
#     #input_text = example["context"] + tokenizer.eos_token
#     text = example["text"]
#     encodings = tokenizer(
#         text,
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )
#     # for causal LM, labels = input_ids
#     encodings["labels"] = encodings["input_ids"].copy()
#     return encodings

# tokenized = dataset.map(
#     preprocess,
#     batched=False,
#     remove_columns=["text"]
# )

# # 4) SET UP TRAINING ARGUMENTS
# training_args = TrainingArguments(
#     output_dir="./fine_tuned_tags_dialoGPT",
#     num_train_epochs=2,             # fewer epochs to start
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     learning_rate=2e-5,             # lower LR
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     logging_steps=50,
# )

# # 5) INITIALIZE & RUN THE TRAINER
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized["train"],
#     eval_dataset=tokenized["test"],
#     tokenizer=tokenizer,
# )

# trainer.train()

# # 6) SAVE YOUR FINE‑TUNED MODEL
# trainer.save_model("./fine_tuned_tags_dialoGPT")
# tokenizer.save_pretrained("./fine_tuned_tags_dialoGPT")
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

# 1) LOAD & SPLIT YOUR DATASET
dataset = load_dataset(
    "json",
    data_files="dialogue_with_tags.jsonl",
    split="train"
).train_test_split(test_size=0.1)

# 2) LOAD MODEL & TOKENIZER
model_name = "microsoft/DialoGPT-small"
tokenizer  = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model      = AutoModelForCausalLM.from_pretrained(model_name)

# 3) PREPROCESS: SPLIT User/Bot, MASK PROMPT TOKENS
def preprocess(example):
    text = example["text"]
    # split at the "\nBot:" marker
    user_part, bot_part = text.split("\nBot:", 1)

    prompt = user_part + "\nBot:"                  
    target = " " + bot_part.strip() + tokenizer.eos_token

    # tokenize the full string so the model sees prompt+target
    enc = tokenizer(
        prompt + target,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    input_ids = enc["input_ids"]

    # compute how many tokens prompt took
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # build labels: -100 for prompt, real token IDs for target
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    enc["labels"] = labels
    return enc

tokenized = dataset.map(
    preprocess,
    batched=False,
    remove_columns=["text"]
)

# 4) SET UP TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./fine_tuned_AGAIN",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
)

# 5) INITIALIZE & RUN THE TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()

# 6) SAVE YOUR FINE‑TUNED MODEL
trainer.save_model("./fine_tuned_Epochs") # how its supposed to feel......
tokenizer.save_pretrained("./fine_tuned_Epochs")

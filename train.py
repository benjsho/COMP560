from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# 1) LOAD & SPLIT YOUR DATASET
dataset = load_dataset(
    "json",
    data_files="EBnpc_dataset.json",
    split="train"
).train_test_split(test_size=0.1)

# 2) LOAD MODEL & TOKENIZER
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3) PREPROCESS: CONCAT CONTEXT + RESPONSE & CREATE LABELS
def preprocess(batch):
    inputs = [
        # Combine context and response, then EOS
        ex["context"] + " " + ex["response"] + tokenizer.eos_token
        for ex in batch
    ]
    enc = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # Labels are the same as input_ids → model learns to reproduce your lines
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 4) SET UP TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./fine_tuned",     # where to save checkpoints
    num_train_epochs=3,            # try 3 epochs to start
    per_device_train_batch_size=4, # adjust to your GPU/CPU
    per_device_eval_batch_size=4,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
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
trainer.save_model("./fine_tuned")
tokenizer.save_pretrained("./fine_tuned")

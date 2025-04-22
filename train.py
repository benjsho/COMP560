from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# 1) LOAD & SPLIT YOUR DATASET
dataset = load_dataset(
    "json",
    data_files="dialogue_with_tags.jsonl",
    split="train"
).train_test_split(test_size=0.1)

# 2) LOAD MODEL & TOKENIZER
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model     = AutoModelForCausalLM.from_pretrained(model_name)

# 3) PREPROCESS: CONCAT CONTEXT + RESPONSE & CREATE LABELS
def preprocess(example):
    # Only input is the context
    #input_text = example["context"] + tokenizer.eos_token
    input_text = "[Context] " + example["instruction"] + "\n[Response]\n"
    target_text = example["response"] + tokenizer.eos_token

    input_encodings = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    input_encodings["labels"] = target_encodings["input_ids"]
    return input_encodings


tokenized = dataset.map(
    preprocess,
    batched=False,
    remove_columns=dataset["train"].column_names
)

# 4) SET UP TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./fine_tuned_dialogue",     # where to save checkpoints
    num_train_epochs=5,            # try 3 epochs to start
    per_device_train_batch_size=4, # adjust to your GPU/CPU
    per_device_eval_batch_size=4,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2
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
trainer.save_model("./fine_tuned_dialogue")
tokenizer.save_pretrained("./fine_tuned_L_dialogue")

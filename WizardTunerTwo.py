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
    data_files="WizardDatasetFormatted.jsonl",  
    split="train"
).train_test_split(test_size=0.1)

# 2) LOAD MODEL & TOKENIZER
model_name = "microsoft/DialoGPT-medium"
tokenizer  = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model      = AutoModelForCausalLM.from_pretrained(model_name)

# 3) (OPTIONAL) INSPECT MAXIMUM LENGTH NEEDED
# Quickly check the longest example to choose a good max_length
def find_max_length(dataset, tokenizer):
    lengths = []
    for example in dataset:
        full_text = example["prompt"] + example["completion"]
        enc = tokenizer(full_text, add_special_tokens=False)
        lengths.append(len(enc["input_ids"]))
    return max(lengths)

# Run once to find a good max_length
max_length = find_max_length(dataset["train"], tokenizer)
print(f"Recommended max_length: {max_length}")

# Cap it for safety (e.g., 512 or lower if GPU memory is small)
max_length = min(max_length, 512) 

# 4) PREPROCESS FOR CAUSAL LM
def preprocess(example):
    prompt = example["prompt"]
    completion = example["completion"].strip() + tokenizer.eos_token

    full_text = prompt + completion
    enc = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    input_ids = enc["input_ids"]

    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    enc["labels"] = labels
    return enc

tokenized = dataset.map(
    preprocess,
    batched=False,
    remove_columns=["prompt", "completion"]
)

# 5) TRAINING ARGS
training_args = TrainingArguments(
    output_dir="./ft_dialoGPT",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    weight_decay=0.01,
    num_train_epochs=2,
    warmup_ratio=0.1,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    lr_scheduler_type="linear",
    fp16=False,
)

# 6) TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()

# 7) SAVE
trainer.save_model("./FTWizardFour")
tokenizer.save_pretrained("./FTWizardFour")

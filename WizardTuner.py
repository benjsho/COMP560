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
    data_files="WizardDatasetCleaned.jsonl",  # <-- actual file path
    split="train"
).train_test_split(test_size=0.1)

# 2) LOAD MODEL & TOKENIZER
# model_name = "gpt2-medium"
model_name = "microsoft/DialoGPT-medium"
tokenizer  = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model      = AutoModelForCausalLM.from_pretrained(model_name)

# 3) PREPROCESS FOR CAUSAL LM
def preprocess(example):
    prompt = "User: " + example["instruction"] + "\nBot:"
    response = " " + example["response"].strip() + tokenizer.eos_token

    full_input = prompt + response
    enc = tokenizer(
        full_input,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    input_ids = enc["input_ids"]

    # Find the token boundary between prompt and response
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    # Label only the response part
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    enc["labels"] = labels
    return enc

tokenized = dataset.map(
    preprocess,
    batched=False,
    remove_columns=["instruction", "response"]
)

# 4) TRAINING ARGS
training_args = TrainingArguments(
    output_dir="./ft_dialoGPT",
    per_device_train_batch_size=4,       # small model→small batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # simulate a larger batch
    learning_rate=5e-6,                  # lower LR for stable updates
    weight_decay=0.01,
    num_train_epochs=2,                  # 4 epochs → revisit data multiple times
    warmup_ratio=0.1,                    # 10% of steps as warmup
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    
    # lr scheduler: linear decay with warmup
    lr_scheduler_type="linear",
    
    # deepspeed or fp16 if you have GPU headroom
    fp16=False,                           
)

# 5) TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()

# 6) SAVE THE MODEL
trainer.save_model("./FTWizardThree")
tokenizer.save_pretrained("./FTWizardThree")

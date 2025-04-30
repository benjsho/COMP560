# old, but i got some use out of this one. 
# this was the program i used when i presented. 
# do_sample = False saved me. 
# i love preparing ! i knew that plant was there .
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your fine-tuned model
model_path = "./FTWizardFour"  # <- replace if saved elsewhere
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Make sure padding token is set
tokenizer.pad_token = tokenizer.eos_token

# Function to chat with the model
def chat_with_bot(user_prompt):
    # Format as training did
    input_text = f"User: {user_prompt}\nBot:"
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    bad_phrases = ["player interacts", "Player interacts", "player", "Player", "interacts", "interactions", "interactes", "good time", "I don't"]
    # bad_words_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
    #     for p in bad_phrases]
    # Generate a response
    output_ids = model.generate(
        input_ids,
        bad_words_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
            for p in bad_phrases],
        attention_mask=attention_mask,
        max_length=60,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, #big
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1,       
        no_repeat_ngram_size=3, 
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract bot's reply
    if "\nBot:" in output_text:
        reply = output_text.split("\nBot:")[-1].strip()
    else:
        reply = output_text.strip()

    print(f"bot: {reply}")

# Example prompts
test_prompts = [
    "player interacts with plastic plant",
    "player interacts with plant in the corner",
    "player interacts with vending machine"
]

for prompt in test_prompts:
    print(f"\nuser: {prompt}")
    chat_with_bot(prompt)

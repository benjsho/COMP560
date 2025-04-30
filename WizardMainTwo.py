from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "./FTWizardThree"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name).to(
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")
             )

# Two exemplar lines from your dataset:
FEW_SHOT = (
    '{"instruction": "Player interacts with dreaming owl", '
    '"response": "I dreamt of a library where books flew away from questions."}\n'
    '{"instruction": "Player interacts with suspicious bush", '
    '"response": "I\'m *definitely* not hiding a goblin. Nope."}\n'
)

def chat_few_shot():
    print("Type 'quit' to exit.")
    while True:
        inst = input("\nPlayer interacts with: ").strip()
        if inst.lower() == "quit":
            break

        # 1) Build prompt with few-shot + your new line prefix
        prefix = (
            FEW_SHOT
            + f'{{"instruction": "Player interacts with {inst}", "response": "'
        )
        input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)

        # 2) Sample
        output_ids = model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 3) Decode & cut off at the first closing quote+brace
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = decoded[len(prefix):]
        raw = completion.split('"}', 1)[0].strip()

        # 4) Capitalize & punctuate
        if raw:
            resp = raw[0].upper() + raw[1:]
            if resp[-1] not in ".!?":
                resp += "."
        else:
            resp = "[no response]"

        print(f"\n{resp}\n")

if __name__ == "__main__":
    chat_few_shot()



# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load your fine-tuned model and tokenizer
# model_name_or_path = "./FTWizardThree"  # Replace with your model path or Huggingface repo name
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# # Function to get a response
# def get_response(instruction, max_new_tokens=60):
#     # Format input if needed (depends how you trained it - assuming simple here)
#     input_text = instruction.strip()
    
#     # Encode input
#     input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

#     # Generate output
#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=tokenizer.eos_token_id,  # Important for DialoGPT
#             do_sample=True,                      # Enable sampling for more creative responses
#             top_p=0.95,                          # Nucleus sampling
#             top_k=50                             # Limit to top-k tokens
#         )
    
#     # Decode the generated text
#     generated_text = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
#     return generated_text.strip()

# # --- Interactive loop ---
# if __name__ == "__main__":
#     print("Talk to the dreaming creatures! (type 'quit' to exit)\n")
#     while True:
#         instruction = input("You: ")
#         if instruction.lower() == "quit":
#             break
#         response = get_response(instruction)
#         print(f"Creature: {response}\n")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your fine-tuned model
model_path = "./FTWizardOne"  # <- replace if saved elsewhere
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Make sure padding token is set
tokenizer.pad_token = tokenizer.eos_token


# Function to chat with the model
def chat_with_bot(scene):
    scene = scene.replace("player interacts with ", "") # strip player interacts boilerplate
    # prompt = (
    #     "You are an NPC speaking to the player. Your replies are colorful, immersive, and sometimes include a short bit of dialogue. "
    #     "Always respond in exactly two sentences.\n\n"
    #     f"Scene: {scene}\n"
    #     "Response:"
    # )
    prompt = f"Instruction: {scene}\nResponse:"

    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    bad_phrases = ["player interacts", "Player interacts", "player", "Player"]
    # bad_words_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
    #                 for p in bad_phrases]
    # Generate a response
    output_ids = model.generate(

        input_ids,
        bad_words_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
                            for p in bad_phrases],
        #do_sample = True,
        attention_mask=attention_mask,
        min_length=input_ids.shape[-1] + 25,   # force at least 25 new tokens
        max_length=input_ids.shape[-1] + 80,   # allow up to 80 new tokens
        length_penalty=0.8,                    # <1 encourages longer outputs
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.5,
         # repetition controls
        repetition_penalty=1.1,       # >1 penalizes tokens that have already appeared 
        no_repeat_ngram_size=3,     #disallows any 3-gram (like a three word phrasse ts) from repeating
        # bad_words_ids=bad_words_ids,
        
    )

    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # # Extract bot's reply
    # if "\nBot:" in output_text:
    #     reply = output_text.split("\nBot:")[-1].strip()
    # else:
    #     reply = output_text.strip()
    # reply = reply.replace("player interacts", "")
    # reply = reply.split('.')[0].strip() + '.'
    # print(f"🧠 Bot: {reply}")
    text  = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = text.split("NPC")[-1].split(":",1)[-1].strip()
    print("🧠 Bot:", reply)

# Example prompts
test_prompts = [
        "A weary knight kneels before a cursed fountain at midnight",
    "A traveling merchant offers you a strange, glowing vial",
    "You disturb a sleeping forest spirit resting on a mossy log",
    "A street performer juggles flaming torches in the crowded square",
    "You approach the rickety bridge guarded by a silent troll",
    "A ghostly bard hums a melancholy tune in the abandoned tavern",
    "You find an ornately carved door with no handle—just a keyhole",
    "A retired mercenary polishes his old, battle-scarred sword",
    "You light a lantern in the depths of a pitch-black cavern",
    "A mischievous imp perches on your shoulder, grinning wickedly",
    "You enter a library where the books whisper as you pass",
    "A blind oracle offers to read your future from spilled tea leaves",
    "You stumble upon a circle of glowing runes on the forest floor",
    "A giant owl tilts its head curiously as you walk beneath its tree",
    "You discover a half-frozen pond with perfectly still, crystal water",
    "Player interacts with a bard",
    "Player interacts with a mysterious map",
]
# test
for s in test_prompts:
    print("\n🎭 Scene:", s)
    chat_with_bot(s)
# for prompt in test_prompts:
#     print(f"\n🗣️ User: {prompt}")
#     chat_with_bot(prompt)








# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load model & tokenizer
# model_path = "./FTWizardOne"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model     = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.eos_token

# # Block “player interacts” if needed
# bad_phrases   = ["player interacts", "Player interacts"]
# bad_words_ids = [tokenizer(phrase, add_special_tokens=False)["input_ids"]
#                  for phrase in bad_phrases]

# def chat_with_bot(scene_description):
#     # Build a single string from your scene description
#     input_text = (
#         "System: You are a witty medieval NPC.\n"
#         f"Scene: {scene_description}\n"
#         "NPC:"
#     )

#     input_ids      = tokenizer.encode(input_text, return_tensors="pt")
#     attention_mask = torch.ones_like(input_ids)

#     output_ids = model.generate(
#         input_ids,
#         attention_mask=attention_mask,
#         max_length=input_ids.shape[-1] + 50,
#         pad_token_id=tokenizer.eos_token_id,

#         # sampling controls
#         do_sample=True,
#         top_k=20,
#         top_p=0.7,
#         temperature=0.4,

#         # repetition & boilerplate controls
#         repetition_penalty=1.4,
#         no_repeat_ngram_size=2,
#         bad_words_ids=bad_words_ids,
#     )

#     text  = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     reply = text.split("NPC:")[-1].strip()
#     # final cleanup
#     reply = reply.replace("player interacts", "").split('.')[0].strip() + '.'
#     return reply

# # A single list of prompts you pass in
# test_prompts = [
#     "A hero greets the talking broom",
#     "The player sits before a cranky college professor",
#     "You sneak up on the sleepy dragon",
#     "A traveler meets the sarcastic owl",
#     "You try to haggle cursed trinkets from the goblin merchant",
# ]

# for scene in test_prompts:
#     print(f"🗣️ Scene: {scene}")
#     print(f"🧠 NPC : {chat_with_bot(scene)}\n")

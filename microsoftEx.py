# old. 
# this example kept flirting with me. 
# i had to put it down. 
# lots of history here though. 
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# ── 1) load & patch@!!!!
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_AGAIN", padding_side="left")
model     = AutoModelForCausalLM.from_pretrained("./fine_tuned_AGAIN")

# make sure model stops on your EOS
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
system_prompt = (
    "System: You are a professional, neutral assistant. "
)
bad = tokenizer(
    ["i like it", "i love you", "i like you"],  # calm down please
    add_special_tokens=False
).input_ids

# ── 2) Generation settings
gen_cfg = GenerationConfig(
    do_sample = True,
    top_k=30,
    # min_p= 0.1,
    top_p=0.6,
    no_repeat_ngram_size=4,
    repetition_penalty = 1.1,
    max_new_tokens=60,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    temperature = 0.1
)
gen_cfg.bad_words_ids  = bad
model.generation_config = gen_cfg
conversation = system_prompt + "\n"
while True:
    user = input(">> User: ")
    conversation += f"User: {user}\nBot:"
    inputs = tokenizer(conversation, return_tensors="pt")
    outputs = model.generate(**inputs)
    reply = tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    print("Bot:", reply)
    conversation += f" {reply}\n"
# ── 3) Chat loop
for _ in range(5):
    user_text = input(">> User: ")

    # build a clear "User: …\nBot:" prompt
    prompt    = f"User: {user_text}\nBot:"
    inputs    = tokenizer(prompt, return_tensors="pt")
    input_len = inputs.input_ids.shape[-1]

    # generate and then strip off the prompt
    outputs   = model.generate(**inputs)
    gen_ids   = outputs[0, input_len:]                      # ← only the NEW tokens
    reply     = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print("Bot:", reply)






# from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer, logging
# import torch

# # ── Optional: silence HF warnings altogether
# logging.set_verbosity_error()

# # ── 1) Load & patch tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_AGAIN")
# #tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_tags_dialoGPT")
# model = AutoModelForCausalLM.from_pretrained(
#     "./fine_tuned_AGAIN",
#     #"./fine_tuned_tags_dialoGPT",
#     pad_token_id=tokenizer.eos_token_id
# )
# gen_cfg = GenerationConfig(
#     pad_token_id=tokenizer.eos_token_id,
#     temperature=0.6,
#     top_k=50,
#     top_p=0.95,
#     no_repeat_ngram_size=2,
#     max_new_tokens=50
# )
# model.generation_config = gen_cfg
# tokenizer.pad_token    = tokenizer.eos_token
# tokenizer.padding_side = "left"

# # ── 4) Enter your chat loop
# chat_history = None
# for _ in range(5):
#     prompt = input(">> User: ") + tokenizer.eos_token
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True)

#     # Generate using the pre‑assigned GenerationConfig
#     outputs = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#     )

#     reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("bot:", reply)












# from transformers import (
#     AutoModelForCausalLM,
#     GenerationConfig,
#     AutoTokenizer,
#     logging
# )
# import torch

# logging.set_verbosity_error()

# # ── 1) Load & patch tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_AGAIN")
# tokenizer.pad_token    = tokenizer.eos_token
# tokenizer.padding_side = "left"

# # ── 2) Load model & gen config
# model = AutoModelForCausalLM.from_pretrained(
#     "./fine_tuned_AGAIN",
#     pad_token_id=tokenizer.eos_token_id
# )
# model.generation_config = GenerationConfig(
#     pad_token_id           = tokenizer.eos_token_id,
#     temperature            = 0.7,
#     top_k                  = 50,
#     top_p                  = 0.95,
#     no_repeat_ngram_size   = 2,
#     max_new_tokens         = 50
# )

# # ── 3) History + mask
# chat_ids   = None
# chat_mask  = None

# for _ in range(5):
#     raw = input(">> User: ").strip()
#     if not raw:
#         continue

#     # Tokenize *once* with a single EOS
#     enc = tokenizer(
#         raw + tokenizer.eos_token,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     )
#     u_ids, u_mask = enc.input_ids, enc.attention_mask

#     # Append user turn to history & mask
#     if chat_ids is None:
#         chat_ids, chat_mask = u_ids, u_mask
#     else:
#         chat_ids  = torch.cat([chat_ids,  u_ids ], dim=-1)
#         chat_mask = torch.cat([chat_mask, u_mask], dim=-1)

#     # Trim to last 512 tokens
#     if chat_ids.shape[-1] > 512:
#         chat_ids  = chat_ids[:, -512:]
#         chat_mask = chat_mask[:, -512:]

#     # Generate (uses model.generation_config automatically)
#     outputs = model.generate(
#         chat_ids,
#         attention_mask=chat_mask,
#     )

#     # Slice off only the newly generated tokens
#     prev_len   = chat_ids.shape[-1]
#     new_ids    = outputs[:, prev_len:]
#     reply_text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

#     # Optional: strip any lingering “User:” or “Bot:” tags
#     reply_text = reply_text.replace("User:", "").replace("Bot:", "").strip()

#     print("bot:", reply_text)

#     # **Now** update history and mask to include the bot’s reply
#     chat_ids   = outputs
#     bot_mask   = torch.ones((1, new_ids.shape[-1]), dtype=chat_mask.dtype)
#     chat_mask  = torch.cat([chat_mask, bot_mask], dim=-1)













# tokenizer.pad_token      = tokenizer.eos_token      # reuse </s> as pad
# tokenizer.padding_side   = "left"                   # left‑pad

# # ── 2) Load model, align its pad_token_id


# chat_history_ids     = None
# chat_attention_mask  = None

# # ── Chat for 5 turns
# for step in range(2):
#     user_text = input(">> User: ") + tokenizer.eos_token

#     # ── 3) Tokenize *with* left padding & get attention mask
#     inputs = tokenizer(
#         user_text,
#         return_tensors="pt",
#         padding=True    # now pads *left* using eos_token_id
#     )
#     input_ids      = inputs.input_ids
#     attn_mask      = inputs.attention_mask

#     # ── 4) Stitch into history
#     if chat_history_ids is None:
#         bot_input_ids      = input_ids
#         bot_attention_mask = attn_mask
#     else:
#         bot_input_ids      = torch.cat([chat_history_ids, input_ids], dim=-1)
#         bot_attention_mask = torch.cat([chat_attention_mask, attn_mask], dim=-1)

#     # ── 5) Generate with sampling + attention mask
#     chat_history_ids = model.generate(
#         bot_input_ids,
#         attention_mask=bot_attention_mask,
#         max_length=bot_input_ids.shape[-1] + 50,  # allow space for reply
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7,
#         no_repeat_ngram_size=2,
#     )
    

#     # ── 6) Save new attention mask (all tokens in history are real tokens or left‑pads)
#     chat_attention_mask = torch.ones_like(chat_history_ids)

#     # ── 7) Decode *only* the new tokens
#     reply = tokenizer.decode(
#         chat_history_ids[:, bot_input_ids.shape[-1]:][0],
#         skip_special_tokens=True
#     )
#     print("bot:", reply)

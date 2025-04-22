from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_AGAIN", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model     = AutoModelForCausalLM.from_pretrained("./fine_tuned_AGAIN")

chat_history_ids = None

while True:
    user = input(">> User: ")
    if user.strip().lower() == "/reset":
        chat_history_ids = None
        print("[Conversation reset]")
        continue

    new_ids = tokenizer.encode(user + tokenizer.eos_token, return_tensors="pt").to(model.device)
    chat_history_ids = new_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_ids], dim=-1)

    # truncate if too long
    if chat_history_ids.size(-1) > 800:
        chat_history_ids = chat_history_ids[:, -800:]

    old_len = chat_history_ids.size(-1)

    chat_history_ids = model.generate(
        chat_history_ids,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # decode only new tokens
    reply_ids = chat_history_ids[:, -50:]  # or compute prompt length dynamically
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    if bot_reply.lower().startswith(user.lower()):
        bot_reply = bot_reply[len(user):].lstrip(" :,.")
    print("Bot:", bot_reply)
    new_tokens = chat_history_ids[0, old_len:]

#old. futile.
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    StoppingCriteria, 
    StoppingCriteriaList
)

# 1) Load
model_path = "./FTWizardOne"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 2) Persona + 5 few-shot examples
SYSTEM = (
    "System: You are a wry, all-knowing medieval NPC. "
    "When given an instruction, you reply in **exactly two sentences** "
    "with vivid detail and occasional dialogue—no meta-Reddit talk.\n\n"
)

FEW_SHOT = """\
Instruction: player interacts with blacksmith
Response: The forge glows like a second sun, and the blacksmith wipes soot from his brow. “What’ll it be—horseshoes or hull repairs?” he asks.

Instruction: player interacts with merchant
Response: Stacks of gleaming trinkets glint behind his stall as he leans forward. “Potions, scrolls, or perhaps some very dubious cheese?”

Instruction: player interacts with cat
Response: The cat stretches luxuriously, nails clicking on stone. “Off you go,” it seems to purr, “I have better naps to attend.”

Instruction: player interacts with plastic plant
Response: Its leaves never wilt, no matter the season, and it collects every dropped coin like a miser. “I’ve seen better foliage on a tapestry,” it whispers.

Instruction: player interacts with barkeep
Response: He slides a frothy mug across the bar without looking up. “Ale’s fresh, gossip’s stale—take your pick.”  

"""

# 3) A custom stopping criteria so we never bleed into the next Instruction
class StopOnInstruction(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.trigger = tokenizer("Instruction:", add_special_tokens=False)["input_ids"]

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        return len(seq) >= len(self.trigger) and seq[-len(self.trigger):] == self.trigger

stop_criteria = StoppingCriteriaList([StopOnInstruction(tokenizer)])

# 4) Generate
def chat_with_bot(instruction: str) -> str:
    prompt = SYSTEM + FEW_SHOT + f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    out = model.generate(
        **inputs,
        do_sample=True,
        top_k=20,
        top_p=0.8,
        temperature=0.6,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        min_length=inputs["input_ids"].shape[-1] + 15,
        max_new_tokens=50,
        length_penalty=0.5,
        stopping_criteria=stop_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Grab only the text after the *last* "Response:"
    reply = full.rsplit("Response:", 1)[-1].strip()

    # Enforce exactly two sentences
    sents = [s.strip() for s in reply.replace("\n", " ").split(".") if s]
    return ". ".join(sents[:2]) + "."

# 5) Test
for scene in [
    "player interacts with plastic plant",
    "player interacts with plant in the corner",
]:
    print(f"\nInstruction: {scene}")
    print("Response:", chat_with_bot(scene))

from transformers import pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
generator("EleutherAI has", do_sample=True, min_length=20)

[{'generated_text': 'EleutherAI has made a commitment to create new software packages for each of its major clients and has'}]
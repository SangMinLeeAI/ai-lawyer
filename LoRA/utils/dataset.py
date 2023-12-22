import datasets
from random import choice

def preprocess_text(batch):
    # print(batch.keys())
    pa = batch['paragraphs'][0]
    qa = choice(pa['qas'])
    # print(qa)
    s = f"<s>[INST]{qa['answer']['clue_text']+' '+qa['question']}[/INST]{qa['answer']['text']}</s>"
    return {"text":s}

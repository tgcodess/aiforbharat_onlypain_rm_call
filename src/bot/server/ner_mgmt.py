'''
Installation :-
pip install torch

STILL IN PROGRESS - COMBINING TOKENS TO WORD IS PENDING
'''

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

## Initialize tokens
tokenizer, model, ner = None, None, None
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

import torch

def get_predictions(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, is_split_into_words=False, truncation=True)
    word_ids = inputs.word_ids()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = logits.argmax(dim=-1)[0].tolist()
    
    labels = [model.config.id2label[pred] for pred in predictions]

    # Reconstruct full words
    word_map = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_map:
            word_map[word_id] = {
                "tokens": [],
                "label": labels[idx]
            }
        word_map[word_id]["tokens"].append(tokens[idx])
    
    # Join subwords & prepare final results
    final_output = []
    for word_id in sorted(word_map.keys()):
        word_pieces = word_map[word_id]["tokens"]
        word = tokenizer.convert_tokens_to_string(word_pieces).replace(" ", "")
        label = word_map[word_id]["label"]
        final_output.append((word, label))

    return final_output


print(get_predictions("मेरा नाम अर्जुन", tokenizer, model))
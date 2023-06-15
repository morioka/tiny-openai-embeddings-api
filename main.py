from functools import lru_cache
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

app = FastAPI()

#

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str, tokenizer_name: str):
    tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_name)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = True
    )
    model.eval()

    return model, tokenizer

def encode(input_text, **args):
    if args['model'] is not None:
        model = args['model']

    if args['tokenizer'] is not None:
        tokenizer = args['tokenizer']
    else:
        tokenizer = model

    model, tokenizer = get_embedding_model(model, tokenizer)

    tokenized_input = tokenizer.tokenize(input_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_input)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        all_encoder_layers = model(tokens_tensor)
    
    embeddings = all_encoder_layers[1][-2].numpy()[0]
    t = np.mean(embeddings, axis=0)
    t = t.reshape(1, -1)

    return t, len(tokenized_input)

BERT_DEFAULT_SETTINGS = {
    "model": "cl-tohoku/bert-base-japanese-whole-word-masking",
    "tokenizer": "cl-tohoku/bert-base-japanese-whole-word-masking"
}

# OpenAI Embeddings API

# curl https://api.openai.com/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "input": "Your text string goes here",
#     "model": "text-embedding-ada-002"
#   }'

# {
#   "data": [
#     {
#       "embedding": [
#         -0.006929283495992422,
#         -0.005336422007530928,
#         ...
#         -4.547132266452536e-05,
#         -0.024047505110502243
#       ],
#       "index": 0,
#       "object": "embedding"
#     }
#   ],
#   "model": "text-embedding-ada-002",
#   "object": "list",
#   "usage": {
#     "prompt_tokens": 5,
#     "total_tokens": 5
#   }
# }

# -----
# copied from https://platform.openai.com/docs/guides/embeddings/what-are-embeddings

class EmbeddingsInput(BaseModel):
    input: str
    model: str

class EmbeddingsOutputData(BaseModel):
    embedding: List[float]
    index: int
    object: str

class EmbeddingsOutputUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingsOutput(BaseModel):
    data: List[EmbeddingsOutputData]
    model: str
    object: str
    usage: EmbeddingsOutputUsage

@app.post('/v1/embeddings', response_model=EmbeddingsOutput)
async def embeddings(data: EmbeddingsInput):

    model = data.model
    input = data.input

    assert model == BERT_DEFAULT_SETTINGS['model']

    embeddings, num_tokens = encode(input_text=input, **BERT_DEFAULT_SETTINGS)
    embedding = embeddings[0].tolist()

    return {
        "data": [
            {
                "embedding": embedding,
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": model,
        "object": "list",
        "usage": {
            "prompt_tokens": num_tokens,
            "total_tokens": num_tokens
        }
    }

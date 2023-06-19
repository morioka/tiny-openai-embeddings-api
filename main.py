from functools import lru_cache
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from model import model as embedding_model

app = FastAPI()

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
    input: str | List[str]
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

    assert model == embedding_model.BERT_DEFAULT_SETTINGS['model']

    embeddings, num_tokens = embedding_model.encode(input_text=input, 
                                                    **embedding_model.BERT_DEFAULT_SETTINGS)

    return {
        "data": [
            {
                "embedding": e, 
                "index": i, 
                "object": "embedding"
            } for i, e in enumerate(embeddings)
        ],
        "model": model,
        "object": "list",
        "usage": {
            "prompt_tokens": num_tokens,
            "total_tokens": num_tokens
        }
    }

from functools import lru_cache
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertJapaneseTokenizer, BertModel
import torch

app = FastAPI()

# https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        num_tokens = 0
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

            num_tokens += sum(sum(i) for i in encoded_input.attention_mask)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings), num_tokens


MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
model = SentenceBertJapanese(MODEL_NAME)

#sentences = ["暴走したAI", "暴走した人工知能"]
#sentence_embeddings = model.encode(sentences, batch_size=8)
#
#print("Sentence embeddings:", sentence_embeddings)


def encode(input_text, **args):
    if type(input_text) == str:
        input_text = [input_text]

    sentence_embeddings, num_tokens = model.encode(input_text, batch_size=8)

    return sentence_embeddings, num_tokens

BERT_DEFAULT_SETTINGS = {
    "model": MODEL_NAME,
    "tokenizer": MODEL_NAME
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

    assert model == BERT_DEFAULT_SETTINGS['model']

    embeddings, num_tokens = encode(input_text=input, **BERT_DEFAULT_SETTINGS)

    return {
        "data": [
            { "embedding": e.tolist(), "index": i, "object": "embedding"} for i, e in enumerate(embeddings)
        ],
        "model": model,
        "object": "list",
        "usage": {
            "prompt_tokens": num_tokens,
            "total_tokens": num_tokens
        }
    }

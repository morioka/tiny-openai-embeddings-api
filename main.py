from typing import List, Annotated
from pydantic import BaseModel

from fastapi import Body, FastAPI

import model as embedding_model

app = FastAPI(
    title="tiny-openai-embeddings-api",
    description="OpenAI Embeddings API-style local server, runnig on FastAPI",
    version="1.0",
)

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

class SupportedModels(BaseModel):
    models: List[str]

# This is not compatible with OpenAI Embeddings API.
@app.get('/v1/embeddings_supported_models', response_model=SupportedModels)
async def supported_models():
    return {
        "models": embedding_model.BERT_DEFAULT_SETTINGS['supported_models']
    } 

@app.post('/v1/embeddings', response_model=EmbeddingsOutput)
async def embeddings(data: Annotated[EmbeddingsInput,
                            Body(
                                openapi_examples={
                                    "sonoisa/sentence-bert":    {
                                        "summary": "sonoisa/sentence-bert",
                                        "description": "sonoisa/sentence-bert を使った例 768次元",
                                        "value": {
                                            "input": "今日はいい天気です。",
                                            "model": "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
                                            }
                                    },
                                    "intfloat/multilingaul-e5":    {
                                        "summary": "intfloat/multilingaul-e5",
                                        "description": "intfloat/multilingaul-e5 を使った例 1024次元",
                                        "value": {
                                            "input": "遠くの山がきれいです。来てよかったです。",
                                            "model": "intfloat/multilingual-e5-large"
                                        }
                                    }
                                }    
                            )]
 ):

    model = data.model
    input = data.input

    assert model in embedding_model.BERT_DEFAULT_SETTINGS['supported_models']

    embeddings, num_tokens = embedding_model.encode(input_text=input,
                                                    pretrained_model_name_or_path=model, 
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

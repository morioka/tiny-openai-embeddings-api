# tiny-openai-embeddings-api

OpenAI Embeddings API-style local server, runnig on FastAPI. 

This API will be compatible with [Embeddings - OpenAI API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).

## Setup
This was built & tested on Python 3.10.8, Ubutu20.04/WSL2 but should also work on Python 3.9+.

```bash
pip install -r requirements.txt
```

or

```bash
docker compose build
```

## Usage

### server
```bash
export PYTHONPATH=.
uvicorn main:app --host 0.0.0.0
```

or

```bash
docker compose up
```

### client

note: Authorization header may be ignored.

example 1: typical usecase, almost identical to OpenAI Embeddings API example

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text string goes here",
    "model": "cl-tohoku/bert-base-japanese-whole-word-masking"
  }'
```

## License

Everything by [morioka](https://github.com/morioka) is licensed under MIT.

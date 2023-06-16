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
    "model": "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
  }'
```

example 2: typical usecase, almost identical to OpenAI Embeddings API example

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Your text string goes here", "Where is your text string?"]
    "model": "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
  }'
```

### download model from huggingface_hub

```bash
python -m download_model --model_id sonoisa/sentence-bert-base-ja-mean-tokens-v2 --local_dir model
```

## License

Everything by [morioka](https://github.com/morioka) is licensed under MIT License.

## TODO

- 完全オフライン化。手元のファインチューニング済モデルを指定できること
  - [HuggingFaceモデルをローカルにダウンロード・シンボリックリンクを無効にする| WonderHorn/ふしぎな角笛](https://wonderhorn.net/programming/hfdownload.html)
  - [Huggingface Transformersのモデルをオフラインで利用する - Qiita](https://qiita.com/suzuki_sh/items/0b43ca942b7294fac16a)
  - 環境変数 `export TRANSFORMERS_OFFLINE=1` でよい?
- モデルはホスト側のディレクトリにあるものをマウントして、外から与えて用いられること
  - 推論手順が独特の場合、そのコードも外から与えられること
  - アプリは /app 以下に配置されている
  - /model または /app/model 以下に配置したいがが、 /app を作る前にマウントしてしまわないか?
- [bert-as-service](https://bert-as-service.readthedocs.io/en/latest/) ([code](https://github.com/jina-ai/clip-as-service/tree/bert-as-service)) のような起動方法をサポートすること
  - 例えば `bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4`
  - 例えば `docker run --runtime nvidia -dit -p 5555:5555 -p 5556:5556 -v $PATH_MODEL:/model -t bert-as-service $NUM_WORKER`

Enjoy!

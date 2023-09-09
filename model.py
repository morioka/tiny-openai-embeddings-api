from transformers import AutoTokenizer, AutoModel, BertJapaneseTokenizer, BertModel
import torch
#import torch.nn.functional as F
from torch import Tensor

# https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, cache_dir="/app/download_model", device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = BertModel.from_pretrained(model_name_or_path, cache_dir="/app/download_model")
        self.model.eval()

        self.model_name_or_path = model_name_or_path

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

# https://huggingface.co/intfloat/multilingual-e5-large
class MultilingualE5:
    def __init__(self, model_name_or_path, cache_dir="/app/download_model", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name_or_path, cache_dir="/app/download_model")
        self.model.eval()

        self.model_name_or_path = model_name_or_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def average_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        num_tokens = 0
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            batch = ["query: " + sentence for sentence in sentences]

            encoded_input = self.tokenizer(batch, max_length=512, padding=True, 
                                           truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)

            sentence_embeddings = self.average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])

            # normalize embeddings
            #embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.extend(sentence_embeddings)

            num_tokens += sum(sum(i) for i in encoded_input.attention_mask)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings), num_tokens

model = None

def encode(input_text, pretrained_model_name_or_path, **args):

    assert pretrained_model_name_or_path in args["supported_models"]

    global model
    if model is not None and model.model_name_or_path != pretrained_model_name_or_path:
        del model
        model = None

    if model is None:
        if pretrained_model_name_or_path in ["sonoisa/sentence-bert-base-ja-mean-tokens-v2"]:
            model = SentenceBertJapanese(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path in ['intfloat/multilingual-e5-large',
                                               'intfloat/multilingual-e5-base',
                                               'intfloat/multilingual-e5-small']:
            model = MultilingualE5(pretrained_model_name_or_path)
        else:
            model = None

    if type(input_text) == str:
        input_text = [input_text]

    sentence_embeddings, num_tokens = model.encode(input_text, batch_size=8)
    sentence_embeddings = [ e.tolist() for e in sentence_embeddings ]

    return sentence_embeddings, num_tokens

BERT_DEFAULT_SETTINGS = {
    "supported_models": [
        "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
        "intfloat/multilingual-e5-large",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-small",
    ]
}

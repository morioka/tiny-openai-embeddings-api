import huggingface_hub
from classopt import classopt, config

@classopt(default_long=True)
class Args:
    model_id: str
    local_dir: str
    local_dir_use_symlinks: bool = config(action="store_true")
 
model_id = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
local_dir = "./cache/model--sonoisa--sentence-bert-base-ja-mean-tokens-v2"
local_dir_use_symlinks = True

args: Args = Args.from_args()
model_id = args.model_id
local_dir = args.local_dir
local_dir_use_symlinks = args.local_dir_use_symlinks

# https://wonderhorn.net/programming/hfdownload.html
download_path = huggingface_hub.snapshot_download(model_id, local_dir=local_dir, local_dir_use_symlinks=local_dir_use_symlinks)
print("model_id: ", model_id)
print("local_dir: ", local_dir)
print("local_dir_use_symlinks: ", local_dir_use_symlinks)
print("download_path: ", download_path)

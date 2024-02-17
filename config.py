from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350, # Độ dài lớn nhất của câu
        "d_model": 512, # Chiều dài Embedding
        "datasource": 'wmt16', # Nguồn data
        "lang_src": "cs",   # Ngôn ngữ ngồn-en
        "lang_tgt": "en",   # Ngôn ngữ đích-it
        "model_folder": "weights",  # Folder chứa các tệp trọng số
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json", # File chứa tokenizer
        "experiment_name": "runs/tmodel" # File chứa các thông tin về mô hình
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
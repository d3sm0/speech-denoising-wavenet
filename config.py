dataset_config = {
    "get_voice": True,
    "in_memory_percentage": 1,
    "noise_only_percent": 0.1,
    "num_classes": 29,
    "path": "../dataset/",
    "max_size": 100,
    "regain": 0.06,
    "sample_rate": 16000,
    "type": "nsdtsea"
}

model_config = {
    "name": "main",
    "encoding": "binary",
    "dilations": 9,
    "lengths": {
        "res": 3,
        "final": [3, 3],
        "skip": 1,
    },
    "depths": {
        "res": 128,
        "skip": 128,
        "final": [2048, 256],
    },
    "num_stacks": 3,
    "target_field_length": 1601,
    "target_padding": 1,
    "lr": 0.001,
}
train_config = {
    "batch_size": 10,
    "early_stopping_patience": 16,
    "num_epochs": 250,
    "num_test_samples": 100,
    "num_train_samples": 1,
    "path": "logs",
    "save_every":1, # epcoches
    "summarize_every":1, # sample
    "verbosity": 1,
    "save_path": "logs/model/",
    "log_path":"logs/train/"
}

class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)

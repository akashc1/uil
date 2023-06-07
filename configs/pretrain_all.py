import importlib
base_config = importlib.import_module('configs.ViT-B-16')


def get_config():
    config = base_config.get_config()

    # pretraining objectives
    config.mae = True
    config.denoise = True
    config.causal = True

    config.epochs = 5
    config.ckpt_interval_epochs = 1

    return config

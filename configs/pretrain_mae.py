from configs import pretrain_all


def get_config():
    config = pretrain_all.get_config()

    # pretraining objectives
    config.denoise = False
    config.causal = False

    return config

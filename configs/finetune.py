import importlib

base_config = importlib.import_module('configs.ViT-B-16')


def get_config():
    config = base_config.get_config()

    # frozen params
    config.freeze_encoder = True

    # finetuning params
    config.lr_warmup_steps = 10
    config.batch_size = 256
    config.epochs = 30

    # classifier params
    config.classifier_hidden_dim = 768
    config.classifier_num_layers = 1
    config.classifier_dropout_rate = 0

    # dataset
    config.train_data = 'nelorth/oxford-flowers'
    config.train_num_samples = 7169
    config.image_key = 'img'
    config.num_classes = 102

    config.num_workers = 32

    # log/eval intervals
    config.log_interval = 5
    config.eval_interval = 10

    config.global_pool = False

    return config

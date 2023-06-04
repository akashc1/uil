import importlib
base_config = importlib.import_module('configs.ViT-B-16')


def get_config():
    config = base_config.get_config()

    # finetuning params
    config.lr_warmup_steps = 10
    config.batch_size = 2048
    config.epochs = 50

    # classifier params
    config.classifier_hidden_dim = 768
    config.classifier_num_layers = 1
    config.classifier_dropout_rate = 0

    # dataset
    config.train_data = 'cifar10'
    config.train_num_samples = 50000
    config.image_key = 'img'
    config.num_classes = 10

    config.num_workers = 8

    # log/eval intervals
    config.log_interval = 5
    config.eval_interval = 10

    config.global_pool = False

    return config

from configs import finetune as base_ft_config


def get_config():
    config = base_ft_config.get_config()

    config.train_data = 'keremberke/pokemon-classification'
    config.image_key = 'image'
    config.label_key = 'labels'
    config.train_num_samples = 4869
    config.num_classes = 150
    config.batch_size = 64
    # config.epochs = 1

    return config

from configs import finetune as base_ft_config


def get_config():
    config = base_ft_config.get_config()

    config.train_data = 'food101'
    config.image_key = 'image'
    config.label_key = 'label'
    config.train_num_samples = 75750
    config.num_classes = 101
    config.batch_size = 256
    # config.epochs = 1

    return config

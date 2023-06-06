from configs import finetune as base_ft_config


def get_config():
    config = base_ft_config.get_config()

    config.train_data = 'nelorth/oxford-flowers'
    config.image_key = 'image'
    config.label_key = 'label'
    config.num_classes = 102
    config.batch_size = 128
    # config.epochs = 1

    return config

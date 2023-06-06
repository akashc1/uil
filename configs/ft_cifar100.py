from configs import finetune as base_ft_config


def get_config():
    config = base_ft_config.get_config()

    config.train_data = 'cifar100'
    config.label_key = 'fine_label'
    config.num_classes = 100

    return config

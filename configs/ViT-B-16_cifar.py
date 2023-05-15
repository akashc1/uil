import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 3e-4
    config.lr_warmup_steps = 100
    config.lr_cosine_decay = True
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.05
    config.batch_size = 128
    config.epochs = 1

    # model
    config.width = 768
    config.layers = 12
    config.heads = 12
    config.image_size = 224
    config.patch_size = 16

    config.mask_ratio = 0.75
    config.decoder_layers=8
    config.decoder_width=512
    config.decoder_heads=16

    config.attn_dropout_rate = 0
    config.dropout_rate = 0

    # dataset
    config.train_data = 'cifar10'
    config.train_num_samples = 50000
    config.image_key = 'img'

    # dataloader
    config.num_workers = 20

    # logging
    config.wandb = False
    config.logging_interval = 1 #50
    config.eval_interval = 500
    config.ckpt_interval = 1000

    return config

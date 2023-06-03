import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # pretraining objectives
    config.mae = False
    config.denoise = True
    config.causal = True

    # optimizer
    config.learning_rate = 3e-4
    config.lr_warmup_steps = 500
    config.lr_cosine_decay = True
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.05
    config.batch_size = 1024
    config.epochs = 5

    # model
    config.width = 768
    config.layers = 12
    config.heads = 12
    config.image_size = 224
    config.patch_size = 16

    config.noise_std = 0.8
    config.mask_ratio = 0.75
    config.decoder_layers = 8
    config.decoder_width = 512
    config.decoder_heads = 16

    config.attn_dropout_rate = 0
    config.dropout_rate = 0

    config.train_data = 'imagenet-1k'
    config.train_num_samples = 1281167
    config.image_key = 'image'

    # dataloader
    config.num_workers = 8

    # logging
    config.wandb = True
    config.logging_interval = 10
    config.eval_interval = 500
    config.ckpt_interval_epochs = 1

    return config

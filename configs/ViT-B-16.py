import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 3e-4
    config.lr_warmup_steps = 500
    config.lr_cosine_decay = True
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.05
    config.batch_size = 2048
    config.epochs = 10

    # model
    config.width = 768
    config.layers = 12
    config.heads = 12
    config.image_size = 224
    config.patch_size = 16

    config.mask_ratio = 0.75
    config.decoder_layers = 8
    config.decoder_width = 512
    config.decoder_heads = 16

    config.attn_dropout_rate = 0
    config.dropout_rate = 0

    # dataset
    # config.train_data = 'pipe:aws s3 cp --endpoint-url=https://17eed60fe0f04546e8689e3c1872641f.r2.cloudflarestorage.com s3://g-datasets/imagenet/imagenet-train-{000000..000256}.tar -'
    # config.train_data = '/mnt/data/imagenet/imagenet-train-{000000..000256}.tar'
    # config.train_data = '/mnt/datasets/imagenet/imagenet-train-{000000..000000}.tar'
    config.train_data = 'imagenet-1k'
    config.train_num_samples = 1281167

    # dataloader
    config.num_workers = 48

    # logging
    config.wandb = False
    config.logging_interval = 5
    config.eval_interval = 500
    config.ckpt_interval = 1000

    return config
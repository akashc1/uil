import functools
import math
import pathlib
import tempfile
import time

import chex
import colorama
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from absl import app
from absl import flags
from absl import logging
from flax.core import frozen_dict
from flax.training import train_state
from flax.training import checkpoints
from jax import lax
from ml_collections import config_flags

from data import image_transform, get_hf_image_dataset
from model import UIL

Fore = colorama.Fore
Style = colorama.Style

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def patchify(imgs, patch_size):
    """
    imgs: (N, H, W, 3)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[1] == imgs.shape[2] and imgs.shape[2] % p == 0
    imgs = imgs.transpose((0, 3, 1, 2))
    h = w = imgs.shape[2] // p
    x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
    x = jnp.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
    return x


def create_learning_rate_fn(config):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.lr_warmup_steps,
    )
    num_batches = config.train_num_samples // config.batch_size
    train_steps = config.epochs * num_batches
    if config.lr_cosine_decay:
        decay_steps = train_steps - config.lr_warmup_steps
        opt_fn = optax.cosine_decay_schedule(
            init_value=config.learning_rate, decay_steps=decay_steps
        )
    else:
        opt_fn = optax.constant_schedule(config.learning_rate)

    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, opt_fn], boundaries=[config.lr_warmup_steps]
    )
    return learning_rate_fn


def create_weight_decay_param_mask(p):
    def filter_fn(param_name):
        # avoid all biases, layer norms, and embeddings
        if (
            param_name.endswith('bias')
            or 'ln' in param_name
            or param_name.endswith('embedding')
        ):
            return False

        # everything else should be fine
        return True

    p = flax.traverse_util.ModelParamTraversal(lambda x, _: filter_fn(x)).update(
        lambda _: True, p
    )
    p = flax.traverse_util.ModelParamTraversal(lambda x, _: not filter_fn(x)).update(
        lambda _: False, p
    )
    return p


def mae_loss(imgs, pred, mask, p):
    target = patchify(imgs, p)
    loss = (pred - target) ** 2
    loss = loss.mean(axis=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss


# @functools.partial(jax.jit, static_argnums=(2,))
def train_step(state, images, config, rng):
    rng = jax.random.fold_in(rng, state.step)
    dropout_rng, mask_rng = jax.random.split(rng)

    def loss_fn(params):
        pred, mask = UIL(**config, deterministic=False).apply(
            {'params': params},
            images, mask_rng,
            rngs={'dropout': dropout_rng},
        )

        loss = mae_loss(images, pred, mask, config['patch_size'])
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, (loss, pred), rng


def train_one_epoch(config, epoch, state, model_config, train_loader, rng):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    learning_rate_fn = create_learning_rate_fn(config)
    world_size = jax.device_count()
    samples_per_epoch = config.train_num_samples
    num_batches_per_epoch = samples_per_epoch // config.batch_size
    sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
    num_samples = 0

    p_train_step = jax.pmap(
        train_step,
        axis_name='batch',
        donate_argnums=(0,),
        static_broadcasted_argnums=(2,),
    )

    # replicate params and rng
    state = flax.jax_utils.replicate(state)
    rng = jax.random.split(rng, num=world_size)

    for i, batch in enumerate(train_loader):
        step = num_batches_per_epoch * epoch + i

        images = batch['image']
        images = images.permute(0, 2, 3, 1).numpy()
        images = jnp.array(images, dtype=jnp.bfloat16)
        num_samples += images.shape[0]

        batch_size_per_device, ragged = divmod(images.shape[0], world_size)
        if ragged:
            msg = "batch size must be divisible by device count, got {} and {}."
            raise ValueError(msg.format(config.batch_size, world_size))

        shape_prefix = (world_size, batch_size_per_device)
        images = images.reshape(shape_prefix + images.shape[1:])

        data_time_m.update(time.time() - end)

        percent_complete = num_samples / samples_per_epoch * 100

        state, (loss, _), rng = p_train_step(state, images, model_config, rng)

        batch_time_m.update(time.time() - end)
        end = time.time()

        if i % config.logging_interval == 0:
            samples_per_second = config.batch_size * world_size / batch_time_m.val
            samples_per_second_per_gpu = config.batch_size / batch_time_m.val
            lr = jax.tree_map(lambda x: x[0], learning_rate_fn(state.step))
            loss = jnp.mean(loss)
            logging.info(
                f"Train Epoch: {0} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/xpu "
                f"LR: {lr.item():5f} Loss: {loss.item():.4f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": lr.item(),
                "mae_loss": loss.item(),
            }

            for name, val in log_data.items():
                name = "train/" + name
                if config.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            batch_time_m.reset()
            data_time_m.reset()

    state = flax.jax_utils.unreplicate(state)
    return state


def train(config):
    rng = jax.random.PRNGKey(config.seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='uil-')
    logging.info(f'workdir: {workdir}')
    if config.wandb:
        wandb.init(project='uil', config=config)

    # setup model and optimizer
    rng, init_rng = jax.random.split(rng)
    model_config = frozen_dict.FrozenDict(
        image_size=config.image_size,
        patch_size=config.patch_size,
        width=config.width,
        layers=config.layers,
        heads=config.heads,
        mask_ratio=config.mask_ratio,
        decoder_layers=config.decoder_layers,
        decoder_width=config.decoder_width,
        decoder_heads=config.decoder_heads,
        dropout_rate=config.dropout_rate,
        attn_dropout_rate=config.attn_dropout_rate,
    )
    model = UIL(**model_config, deterministic=True)
    fake_img = jnp.ones([2, config.image_size, config.image_size, 3], dtype=jnp.bfloat16)
    params = model.init(init_rng, fake_img, init_rng)['params']

    learning_rate_fn = create_learning_rate_fn(config)
    tx = optax.adamw(
        learning_rate_fn,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay,
        mask=create_weight_decay_param_mask,
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    ckpt_dir = pathlib.Path(workdir) / 'checkpoints'
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    tabulate_fn = nn.tabulate(model, tabulate_rng)
    logging.info(tabulate_fn(fake_img, tabulate_rng))

    # data
    preprocess_train = image_transform(config.image_size, is_train=True)
    train_loader = get_hf_image_dataset(
        data=config.train_data,
        preprocess_fn=preprocess_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    for epoch in range(config.epochs):
        state = train_one_epoch(config, epoch, state, model_config, train_loader, rng)
        checkpoints.save_checkpoint(
            ckpt_dir, state, epoch, keep=float('inf')
        )

    return state


def main(argv):
    del argv  # Unused.

    config = FLAGS.config
    np.random.seed(config.seed)
    _ = train(config)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
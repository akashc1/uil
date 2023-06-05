import math
import pathlib
import tempfile
import time

from absl import app, flags, logging
import colorama
import flax
from flax import traverse_util
from flax.core import frozen_dict
from flax.core.frozen_dict import freeze
import flax.linen as nn
from flax.training import checkpoints, train_state
import jax
from jax import lax
import jax.numpy as jnp
from ml_collections import ConfigDict, config_flags
import numpy as np
import optax
from torch.utils.data import DataLoader
import yaml

from data import get_hf_image_dataset, image_transform
from evaluate import batch_accuracy, eval_dataset
from model import UIL, UILClassifier
import wandb

Fore = colorama.Fore
Style = colorama.Style

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('ckpt_init_path', None, 'Path to pretrained weights', short_name='i')
config_flags.DEFINE_config_file(
    'config',
    'configs/finetune.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True,
    short_name='c',
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


def build_optimizer_tx(config, params):
    learning_rate_fn = create_learning_rate_fn(config)
    adam = optax.adamw(
        learning_rate_fn,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay,
        mask=create_weight_decay_param_mask,
    )
    if not config.freeze_encoder:
        return adam

    optims = {'adam': adam, 'zero': optax.set_to_zero()}
    mask = freeze(
        traverse_util.path_aware_map(
            lambda path, _: 'zero' if 'encoder' in path else 'adam',
            params,
        )
    )
    return optax.multi_transform(optims, mask)


def ce_logits_loss(labels, logits):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def train_step(state, images, labels, rng):
    rng = jax.random.fold_in(rng, state.step)
    dropout_rng, mask_rng = jax.random.split(rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            images, mask_rng,
            rngs={'dropout': dropout_rng},
        )
        loss = ce_logits_loss(labels, logits)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, (loss, pred), rng


def train_one_epoch(
    config: ConfigDict,
    epoch: int,
    state: train_state.TrainState,
    train_loader: DataLoader,
    test_loader: DataLoader,
    rng, jnp.ndarray,
):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    world_size = jax.device_count()
    samples_per_epoch = train_loader.num_samples
    num_batches_per_epoch = train_loader.num_batches

    sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
    num_samples = 0

    test_loader_iter = iter(test_loader)

    learning_rate_fn = create_learning_rate_fn(config)
    p_train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

    # replicate params and rng
    state = flax.jax_utils.replicate(state)
    rng = jax.random.split(rng, num=world_size)

    for i, batch in enumerate(train_loader):
        step = num_batches_per_epoch * epoch + i

        images = batch['image']
        images = images.permute(0, 2, 3, 1).numpy()
        images = jnp.array(images, dtype=jnp.bfloat16)
        labels = jnp.array(batch['label'].numpy(), dtype=jnp.uint8)
        num_samples += images.shape[0]

        batch_size_per_device, ragged = divmod(images.shape[0], world_size)
        if ragged:
            msg = "batch size must be divisible by device count, got {} and {}."
            raise ValueError(msg.format(config.batch_size, world_size))

        shape_prefix = (world_size, batch_size_per_device)
        images = images.reshape(shape_prefix + images.shape[1:])
        labels = labels.reshape(shape_prefix)

        data_time_m.update(time.time() - end)

        percent_complete = num_samples / samples_per_epoch * 100

        state, (loss, logits), rng = p_train_step(state, images, labels, rng)
        loss = jax.tree_map(lambda x: x[0], loss)

        batch_time_m.update(time.time() - end)
        end = time.time()

        if i % config.logging_interval == 0:
            samples_per_second = config.batch_size * world_size / batch_time_m.val
            samples_per_second_per_gpu = config.batch_size / batch_time_m.val
            lr = learning_rate_fn(flax.jax_utils.unreplicate(state.step))

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/xpu "
                f"LR: {lr:5f} Loss: {loss.item():.4f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": lr.item(),
                "cls_loss": loss.item(),
            }

            for name, val in log_data.items():
                if config.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            batch_time_m.reset()
            data_time_m.reset()

        if i % config.eval_interval == 0:
            # training accuracy using the logits we just computed
            tr_acc = batch_accuracy(logits, labels)

            # sample a test batch to forward
            dedup_state = flax.jax_utils.unreplicate(state)
            single_rng = jax.random.fold_in(rng[0], state.step[0])
            dropout_rng, mask_rng = jax.random.split(single_rng)

            test_batch = next(test_loader_iter)
            images = test_batch['image'].permute(0, 2, 3, 1).numpy()
            images = np.array(images, dtype=jnp.bfloat16)
            labels = jnp.array(test_batch['label'].numpy(), dtype=jnp.uint8)

            logits = dedup_state.apply_fn(
                {'params': dedup_state.params},
                images,
                mask_rng,
                rngs={'dropout': dropout_rng},
            )
            ts_acc = batch_accuracy(logits, labels)

            del logits, test_batch

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} "
                f"({percent_complete:.0f}%)] "
                f"step {step} train accuracy: {tr_acc:.4f} test accuracy: {ts_acc:.4f}"
            )

            log_data = {
                "train_acc": tr_acc,
                "test_acc": ts_acc,
            }

            for name, val in log_data.items():
                if config.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

        del images, labels

    state = flax.jax_utils.unreplicate(state)
    return state


def train(config):
    rng = jax.random.PRNGKey(config.seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='uil-')

    workdir = pathlib.Path(workdir)
    logging.info(f'workdir: {str(workdir)}')

    # save training config for sanity
    with open(workdir / 'config.yaml', 'w') as fh:
        yaml.safe_dump(dict(config), fh)

    if config.wandb:
        wandb.init(project='UIL', entity='uil', config=config, name=f'finetune_{workdir.name}')

    # setup model and optimizer
    rng, init_rng = jax.random.split(rng)
    uil_config = frozen_dict.FrozenDict(
        image_size=config.image_size,
        patch_size=config.patch_size,
        width=config.width,
        layers=config.layers,
        heads=config.heads,
        noise_std=config.noise_std,
        mask_ratio=config.mask_ratio,
        decoder_layers=config.decoder_layers,
        decoder_width=config.decoder_width,
        decoder_heads=config.decoder_heads,
        dropout_rate=config.dropout_rate,
        attn_dropout_rate=config.attn_dropout_rate,
        do_mae=False,
        do_denoise=False,
        do_causal=False,
    )
    mae_model = UIL(**uil_config, deterministic=True)

    classifier_config = frozen_dict.FrozenDict(
        encoder=mae_model,
        hidden_dim=config.classifier_hidden_dim,
        n_layers=config.classifier_num_layers,
        dropout_rate=config.classifier_dropout_rate,
        deterministic=True,
        n_classes=config.num_classes,
        global_pool=config.global_pool,
    )
    classifier = UILClassifier(**classifier_config)

    fake_img = jnp.ones([2, config.image_size, config.image_size, 3], dtype=jnp.bfloat16)
    params = classifier.init(init_rng, fake_img, init_rng)['params']
    if FLAGS.ckpt_init_path is not None:
        params = classifier.insert_pretrained_params(FLAGS.ckpt_init_path, params)
        logging.info(f"Loaded pretrained weights from {FLAGS.ckpt_init_path}")
    else:
        logging.info("No pretrained weights provided -- training from scratch!!")

    tx = build_optimizer_tx(config, params)
    state = train_state.TrainState.create(apply_fn=classifier.apply, params=params, tx=tx)
    ckpt_dir = workdir / 'checkpoints'

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    tabulate_fn = nn.tabulate(classifier, tabulate_rng)
    logging.info(tabulate_fn(fake_img, tabulate_rng))

    # data
    train_loader = get_hf_image_dataset(
        data=config.train_data,
        split='train',
        preprocess_fn=image_transform(config.image_size, is_train=True),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_key=config.image_key,
    )
    test_loader = get_hf_image_dataset(
        data=config.train_data,
        split='test',
        preprocess_fn=image_transform(config.image_size, is_train=False),
        batch_size=config.batch_size // jax.device_count(),  # Just using one device for eval
        num_workers=config.num_workers // jax.device_count(),
        image_key=config.image_key,
    )

    for epoch in range(config.epochs):
        state = train_one_epoch(
            config,
            epoch,
            state,
            train_loader,
            test_loader,
            rng,
        )

        if (epoch + 1) % config.ckpt_interval_epochs == 0 or epoch == config.epochs - 1:
            checkpoints.save_checkpoint(
                ckpt_dir, state, epoch, keep=float('inf')
            )

    return state


def main(argv):
    del argv  # Unused.

    config = FLAGS.config
    np.random.seed(config.seed)
    final_state = train(config)

    # run evaluation on the full test set
    test_loader = get_hf_image_dataset(
        data=config.train_data,
        split='test',
        preprocess_fn=image_transform(config.image_size, is_train=False),
        batch_size=config.batch_size // jax.device_count(),  # Just using one device for eval
        num_workers=config.num_workers // jax.device_count(),
        image_key=config.image_key,
    )
    test_accuracy = eval_dataset(test_loader, final_state)
    if config.wandb:
        wandb.log({'final_test_accuracy': test_accuracy})
        logging.info(Fore.MAGENTA + Style.BRIGHT + "Final test accuracy: " + str(test_accuracy))


if __name__ == '__main__':
    jax.config.config_with_absl()
    app.run(main)

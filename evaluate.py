import flax
import jax
from jax import numpy as jnp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def batch_accuracy(logits, labels):
    preds = logits.argmax(-1)
    return (preds == labels).mean()


def eval_batch(images, labels, state, rng):
    logits = state.apply_fn(
        {'params': state.params},
        images, rng,
    )
    acc = batch_accuracy(logits, labels)
    return acc


def eval_dataset(dl: DataLoader, state):

    rng = jax.random.PRNGKey(0)  # isn't be used at all, but must pass in to forward pass
    world_size = jax.device_count()
    bs = dl.batch_size
    bs_device = bs // world_size
    shape_prefix = (world_size, bs_device)

    state = flax.jax_utils.replicate(state)
    rng = jax.random.split(rng, num=world_size)
    p_eval_batch = jax.pmap(eval_batch, axis_name='batch', donate_argnums=(0, 1))

    acc_sum = 0
    for batch in tqdm(dl, desc='Evaluating dataset'):
        images, labels = batch['image'].permute(0, 2, 3, 1).numpy(), batch['label'].numpy()
        images = jnp.array(images, dtype=jnp.bfloat16).reshape(shape_prefix + images.shape[1:])
        labels = jnp.array(labels, dtype=jnp.uint8).reshape(shape_prefix)

        acc_sum += p_eval_batch(images, labels, state, rng).mean()

    accuracy = acc_sum / len(dl)

    return accuracy

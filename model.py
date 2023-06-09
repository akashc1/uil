from flax.core import frozen_dict
from pathlib import Path
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp


def unbatched_gather(x, ids_keep):
    return x[ids_keep, Ellipsis]


batched_gather = jax.vmap(unbatched_gather)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.bfloat16)
    grid_w = jnp.arange(grid_size, dtype=jnp.bfloat16)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = jnp.concatenate([jnp.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.bfloat16)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Block(nn.Module):
    d_model: int
    heads: int
    dropout_rate: float
    attn_dropout_rate: float
    deterministic: bool
    mlp_ratio: float = 4.0

    def setup(self):
        self.ln1 = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.ln2 = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.attention = nn.SelfAttention(
            num_heads=self.heads,
            dropout_rate=self.attn_dropout_rate,
            deterministic=self.deterministic,
            param_dtype=jnp.bfloat16,
        )

        mlp_width = int(self.mlp_ratio * self.d_model)
        self.mlp = nn.Sequential(
            [
                nn.Dense(mlp_width, param_dtype=jnp.bfloat16),
                nn.gelu,
                nn.Dense(self.d_model, param_dtype=jnp.bfloat16),
                nn.Dropout(
                    self.dropout_rate,
                    deterministic=self.deterministic
                ),
            ]
        )

    def __call__(self, x, attn_mask=None):
        x = x + self.attention(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    num_layers: int
    block_params: dict

    @nn.compact
    def __call__(self, x, attn_mask=None):
        for i in range(self.num_layers):
            x = Block(**{**self.block_params, 'name': f'block{i}'})(x, attn_mask)

        return x


class UIL(nn.Module):
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int

    do_causal: bool
    do_mae: bool
    do_denoise: bool

    noise_std: float
    mask_ratio: float
    decoder_layers: int
    decoder_width: int
    decoder_heads: int

    deterministic: bool
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    def setup(self):
        block_params = {
            'd_model': self.width,
            'heads': self.heads,
            'attn_dropout_rate': self.attn_dropout_rate,
            'dropout_rate': self.dropout_rate,
            'deterministic': self.deterministic,
        }
        decoder_block_params = {
            **block_params,
            'd_model': self.decoder_width,
            'heads': self.decoder_heads,
        }
        # Encoder
        self.conv1 = nn.Conv(
            self.width,
            (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=False,
            param_dtype=jnp.bfloat16,
        )
        grid_size = self.image_size // self.patch_size
        self.positional_embedding = self.param(
            'pos_embed',
            lambda _, shape: get_2d_sincos_pos_embed(shape[1], shape[0], cls_token=True),
            (grid_size, self.width),
        ).astype(jnp.bfloat16)
        self.class_embedding = self.param(
            'class_embed',
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.width)),
            (self.width,),
        ).astype(jnp.bfloat16)
        self.transformer = Transformer(self.layers, block_params)

        self.ln_pre = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.ln_post = nn.LayerNorm(param_dtype=jnp.bfloat16)

        # MAE Decoder
        self.decoder_embed = nn.Dense(self.decoder_width, use_bias=True, param_dtype=jnp.bfloat16)
        self.mask_token = self.param(
            'mask_token',
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.decoder_width)),
            (self.decoder_width,),
        ).astype(jnp.bfloat16)
        self.decoder_positional_embedding = self.param(
            'dec_pos_embed',
            lambda _, shape: get_2d_sincos_pos_embed(shape[1], shape[0], cls_token=True),
            (grid_size, self.decoder_width),
        )
        self.decoder = Transformer(self.decoder_layers, decoder_block_params)
        self.dec_ln_pre = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.dec_ln_post = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.decoder_pred = nn.Dense(self.patch_size**2 * 3, use_bias=True, param_dtype=jnp.bfloat16)

        # Denoising decoder
        self.denoise_decoder_embed = nn.Dense(self.decoder_width, use_bias=True, param_dtype=jnp.bfloat16)
        self.denoise_decoder_positional_embedding = self.param(
            'denoise_dec_pos_embed',
            lambda _, shape: get_2d_sincos_pos_embed(shape[1], shape[0], cls_token=True),
            (grid_size, self.decoder_width),
        )
        self.denoise_decoder = Transformer(self.decoder_layers, decoder_block_params)
        self.denoise_dec_ln_pre = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.denoise_dec_ln_post = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.noise_pred = nn.Dense(self.patch_size**2 * 3, use_bias=True, param_dtype=jnp.bfloat16)

    def fwd_transformer(self, x, attn_mask=None):
        """Assumes that positional embeddings were previously added"""

        cls_token = (
            self.class_embedding.astype(x.dtype)
            + self.positional_embedding[:1, :].astype(x.dtype)
        )
        cls_token = jnp.broadcast_to(cls_token, (x.shape[0], 1, self.width))
        x = jnp.concatenate([cls_token, x], axis=1)

        x = self.ln_pre(x)
        x = self.transformer(x, attn_mask)
        x = self.ln_post(x)
        return x

    def random_masking(self, x, mask_ratio, rng):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        rng, key = jax.random.split(rng)
        noise = jax.random.uniform(key, shape=(N, L), dtype=x.dtype)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = batched_gather(x, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones((N, L), dtype=x.dtype)
        mask = mask.at[:, :len_keep].set(0)
        # unshuffle to get the binary mask
        mask = batched_gather(mask, ids_restore)
        return x_masked, mask, ids_restore

    def encode_mae(self, x, mask_ratio, rng):
        x = self.conv1(x)  # [*, grid, grid, width]
        x = x.reshape((x.shape[0], -1, x.shape[-1]))  # [*, grid ** 2, width]

        x = x + self.positional_embedding[1:, :].astype(x.dtype)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, rng)
        x = self.fwd_transformer(x)

        return x, mask, ids_restore

    def decode_mae(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = jnp.tile(
            self.mask_token,
            (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        )
        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = batched_gather(x_, ids_restore)
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)

        x = x + self.decoder_positional_embedding.astype(x.dtype)

        x = self.dec_ln_pre(x)
        x = self.decoder(x)
        x = self.dec_ln_post(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def encode_causal(self, x, mask_ratio):
        x = self.conv1(x)
        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        T = x.shape[1]
        x = x + self.positional_embedding[1:, :].astype(x.dtype)

        len_keep = int(x.shape[1] * (1 - mask_ratio))
        x = x[:, :len_keep]

        x = self.fwd_transformer(x)
        return x, T

    def decode_causal(self, x, T):
        x = self.decoder_embed(x)
        mask_tokens = jnp.tile(
            self.mask_token,
            (x.shape[0], T - x.shape[1] + 1, 1)
        )
        x = jnp.concatenate([x, mask_tokens], axis=1)

        x = x + self.decoder_positional_embedding.astype(x.dtype)
        x = self.dec_ln_pre(x)
        attn_mask = nn.make_causal_mask(jnp.ones((x.shape[0], T + 1), dtype=x.dtype))
        x = self.decoder(x, attn_mask)
        x = self.dec_ln_post(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def perturb(self, x, noise_std, rng):
        noise = noise_std * jax.random.normal(rng, shape=x.shape, dtype=x.dtype)
        x = x + noise

        x = self.conv1(x)  # [*, grid, grid, width]
        x = x.reshape((x.shape[0], -1, x.shape[-1]))  # [*, grid ** 2, width]

        x = x + self.positional_embedding[1:, :].astype(x.dtype)
        x = self.fwd_transformer(x)

        return x, noise

    def denoise(self, x):
        x = self.denoise_decoder_embed(x)

        x = x + self.denoise_decoder_positional_embedding.astype(x.dtype)

        x = self.denoise_dec_ln_pre(x)
        x = self.denoise_decoder(x)
        x = self.denoise_dec_ln_post(x)
        x = self.noise_pred(x)
        x = x[:, 1:, :]
        return x

    def __call__(self, x, rng):
        rng_denoise, rng_mae = jax.random.split(rng)
        pred_mae, pred_causal, mask, pred_noise, true_noise = (None,) * 5

        # Denoising
        if self.do_denoise:
            x_perturb, true_noise = self.perturb(x, self.noise_std, rng_denoise)
            pred_noise = self.denoise(x_perturb)

        # MAE
        if self.do_mae:
            latent, mask, ids_restore = self.encode_mae(x, self.mask_ratio, rng_mae)
            pred_mae = self.decode_mae(latent, ids_restore)

        # Causal
        if self.do_causal:
            latent_causal, T = self.encode_causal(x, self.mask_ratio)
            pred_causal = self.decode_causal(latent_causal, T)
        return pred_mae, pred_causal, mask, pred_noise, true_noise


class UILClassifier(nn.Module):
    encoder: nn.Module

    hidden_dim: int
    n_layers: int
    deterministic: bool

    n_classes: int

    dropout_rate: float = 0.0
    global_pool: bool = False

    def setup(self):
        mlp_layers = sum(
            (
                [
                    nn.Dense(self.hidden_dim, param_dtype=jnp.bfloat16),
                    nn.gelu,
                    nn.Dropout(
                        self.dropout_rate,
                        deterministic=self.deterministic
                    ),
                ]
                for _ in range(self.n_layers)
            ),
            start=[]
        )
        self.mlp = nn.Sequential(mlp_layers)
        self.class_head = nn.Dense(self.n_classes)

    @staticmethod
    def insert_pretrained_params(ckpt_path, params: frozen_dict.FrozenDict):
        # Parallel loading seems broken
        ckpt_path = Path(ckpt_path)
        if ckpt_path.name != 'checkpoints':
            ckpt_path /= 'checkpoints'
        pretrained_params = checkpoints.restore_checkpoint(ckpt_path, None, parallel=False)['params']
        params = params.unfreeze()
        params['encoder'] = pretrained_params
        return frozen_dict.freeze(params)

    def __call__(self, x, rng):
        x_encoded, _, _ = self.encoder.encode_mae(x, 0.0, rng)
        x_encoded = x_encoded[:, 1:].mean(1) if self.global_pool else x_encoded[:, 0]
        logits = self.class_head(self.mlp(x_encoded))
        return logits

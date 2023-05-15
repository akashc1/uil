import flax.linen as nn
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

    emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
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

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

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
        # B, T, _ = x.shape
        # # causal_mask = nn.make_causal_mask(jnp.ones((B, T)))
        x = x + self.attention(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class UIL(nn.Module):
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int

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
        self.transformer = nn.Sequential([
            Block(
                d_model=self.width,
                heads=self.heads,
                attn_dropout_rate=self.attn_dropout_rate,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
            )
            for _ in range(self.layers)
        ])

        self.ln_pre = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.ln_post = nn.LayerNorm(param_dtype=jnp.bfloat16)

        # Decoder
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
        self.decoder = nn.Sequential([
            Block(
                d_model=self.decoder_width,
                heads=self.decoder_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
            )
            for _ in range(self.decoder_layers)
        ])
        self.dec_ln_pre = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.dec_ln_post = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.decoder_pred = nn.Dense(self.patch_size**2 * 3, use_bias=True, param_dtype=jnp.bfloat16) # decoder to patch

        self.noise_pred = nn.Dense(self.patch_size**2 * 3, use_bias=True, param_dtype=jnp.bfloat16)

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

        cls_token = self.class_embedding.astype(x.dtype) + \
                    self.positional_embedding[:1, :].astype(x.dtype)
        cls_token = jnp.broadcast_to(cls_token, (x.shape[0], 1, self.width))
        x = jnp.concatenate([cls_token, x], axis=1)

        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)

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

    def perturb(self, x, noise_std, rng):
        noise = jax.random.normal(rng, shape=x.shape, dtype=x.dtype)
        x = x + noise_std * noise

        x = self.conv1(x)  # [*, grid, grid, width]
        x = x.reshape((x.shape[0], -1, x.shape[-1]))  # [*, grid ** 2, width]

        x = x + self.positional_embedding[1:, :].astype(x.dtype)
        cls_token = self.class_embedding.astype(x.dtype) + \
                    self.positional_embedding[:1, :].astype(x.dtype)
        cls_token = jnp.broadcast_to(cls_token, (x.shape[0], 1, self.width))
        x = jnp.concatenate([cls_token, x], axis=1)

        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)

        return x, noise

    def denoise(self, x):
        x = self.decoder_embed(x)

        x = x + self.decoder_positional_embedding.astype(x.dtype)

        x = self.dec_ln_pre(x)
        x = self.decoder(x)
        x = self.dec_ln_post(x)
        x = self.noise_pred(x)
        x = x[:, 1:, :]
        return x

    def __call__(self, x, rng):
        rng_denoise, rng_mae = jax.random.split(rng)

        # Denoising
        x_perturb, true_noise = self.perturb(x, self.noise_std, rng_denoise)
        pred_noise = self.denoise(x_perturb)

        # MAE
        latent, mask, ids_restore = self.encode_mae(x, self.mask_ratio, rng_mae)
        pred_mae = self.decode_mae(latent, ids_restore)
        return pred_mae, mask, pred_noise, true_noise

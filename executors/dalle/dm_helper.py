import time

time.sleep(20)  # let diffusion load first to avoid the surge competition on GPU memory, which results in OOM

import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from PIL import Image
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from vqgan_jax.modeling_flax_vqgan import VQModel

# # dalle-mini
# DALLE_MODEL = "dalle-mini/dalle-mini/kvwti2c9:latest"
# dtype = jnp.float32
#
# # dalle-mega
# DALLE_MODEL = 'dalle-mini/dalle-mini/mega-1:latest'
# dtype=jnp.float32

# dall-mega-fp16
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
dtype=jnp.float16

DALLE_COMMIT_ID = None

VQGAN_REPO = 'dalle-mini/vqgan_imagenet_f16_16384'
VQGAN_COMMIT_ID = 'e93a26e7707683d349bf5d5c41c5b0ef69b677a9'

gen_top_k = None
gen_top_p = 0.9
temperature = None
cond_scale = 3.0

wandb.init(anonymous='must')

# Load models & tokenizer

model, params = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=dtype, _do_init=False)
vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, dtype=jnp.float32, _do_init=False)

print('device count:', jax.device_count())
params = replicate(params)
vqgan_params = replicate(vqgan_params)


# model inference
@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode images
@partial(jax.pmap, axis_name='batch')
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)


processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)


def tokenize_prompt(prompt: str):
    tokenized_prompt = processor([prompt])
    return replicate(tokenized_prompt)


def generate_images(prompt: str, num_predictions: int):
    tokenized_prompt = tokenize_prompt(prompt)

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    # generate images
    images = []
    for i in range(max(1, num_predictions // jax.device_count())):
        # get a new key
        key, subkey = jax.random.split(key)

        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )

        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]

        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for img in decoded_images:
            images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))

    return images

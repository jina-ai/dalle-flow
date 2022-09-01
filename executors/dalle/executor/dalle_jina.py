import random
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from vqgan_jax.modeling_flax_vqgan import VQModel


class Dalle:
    def __init__(self):

        DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"

        DALLE_COMMIT_ID = None

        VQGAN_REPO = 'dalle-mini/vqgan_imagenet_f16_16384'
        VQGAN_COMMIT_ID = 'e93a26e7707683d349bf5d5c41c5b0ef69b677a9'

        # Load models & tokenizer

        self.model, self.params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, dtype=jnp.float32, _do_init=False
        )
        self.processor = DalleBartProcessor.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID
        )

    # model inference
    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        self, tokenized_prompt, key, top_k, top_p, temperature, condition_scale
    ):

        return self.model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=self.params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode images
    @partial(jax.pmap, axis_name='batch')
    def p_decode(self, indices):
        return self.vqgan.decode_code(indices, params=self.vqgan_params)

    def tokenize_prompt(self, prompt: str):
        tokenized_prompt = self.processor([prompt])
        return replicate(tokenized_prompt)

    def generate_images(self, prompt: str, num_predictions: int):
        tokenized_prompt = self.tokenize_prompt(prompt)

        # create a random key
        seed = random.randint(0, 2 ** 32 - 1)
        key = jax.random.PRNGKey(seed)

        # generate images
        images = []
        for i in range(max(1, num_predictions // jax.device_count())):
            # get a new key
            key, subkey = jax.random.split(key)

            gen_top_k = None
            gen_top_p = 0.9
            temperature = None
            cond_scale = 10.0

            # generate images
            encoded_images = self.generate(
                tokenized_prompt,
                prng_key=shard_prng_key(subkey),
                top_k=gen_top_k,
                top_p=gen_top_p,
                temperature=temperature,
                condition_scale=cond_scale,
            )

            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]

            # decode images
            decoded_images = self.p_decode(encoded_images)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for img in decoded_images:
                images.append(np.asarray(img * 255, dtype=np.uint8))

        return images

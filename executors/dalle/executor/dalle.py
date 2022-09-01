import time
from io import BytesIO
from typing import Dict

from jina import Executor, requests, DocumentArray, Document

from . import dm_helper
from dalle_jina import *

class DalleGenerator(Executor):

    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        DALLE_MODEL = 'dalle-mini/dalle-mini/mega-1-fp16:latest'  # DALLE_MODEL = DALLE_MODEL_PATH
        VQGAN_REPO = 'dalle-mini/vqgan_imagenet_f16_16384'  # VQGAN_REPO = VQGAN_REPO_PATH

        # Load models & tokenizer
        self.model, self.params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, dtype=jnp.float32, _do_init=False
        )

        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

        self.model =


    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):
        num_images = int(parameters.get('num_images', 1))
        for d in docs:
            generated_imgs = self.model.generate_images(d.text, num_images)

            for img in generated_imgs:
                _d = Document(
                    mime_type='image/png',
                    tags={  # TODO do we need these?
                        'text': d.text,
                        'generator': 'DALLE-mega',
                    },
                    tensor=img
                ).convert_tensor_to_uri()
                _d.text = d.text  # TODO do we need this?
                d.matches.append(_d)

    def generate(self, prompt, num_predictions):
        tokenized_prompt = tokenize_prompt(prompt)

        # create a random key
        seed = random.randint(0, 2 ** 32 - 1)
        key = jax.random.PRNGKey(seed)

        # generate images
        images = []
        for i in range(max(1, num_predictions // jax.device_count())):
            # get a new key
            key, subkey = jax.random.split(key)

            # generate images
            encoded_images = model.generate(
                tokenized_prompt,
                prng_key=shard_prng_key(subkey),
                params=params,
                top_k=gen_top_k,
                top_p=gen_top_p,
                temperature=temperature,
                condition_scale=cond_scale,
            )

            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]

            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for img in decoded_images:
                images.append(np.asarray(img * 255, dtype=np.uint8))


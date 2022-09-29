import base64
import time
import torch

from io import BytesIO
from operator import itemgetter
from random import randint
from typing import Dict
from urllib.request import urlopen

import numpy as np

from PIL import Image
from stable_inference import StableDiffusionInference
from stable_inference.util import (
    combine_weighted_subprompts,
    slerp,
)

from tqdm import trange

from jina import Executor, DocumentArray, Document, requests


K_DIFF_SAMPLERS = {'k_lms', 'dpm2', 'dpm2_ancestral', 'heun',
    'euler', 'euler_ancestral'}


MAX_STEPS = 250
MIN_HEIGHT = 384
MIN_WIDTH = 384


def document_to_pil(doc):
    uri_data = urlopen(doc.uri)
    return Image.open(BytesIO(uri_data.read()))


class StableDiffusionGenerator(Executor):
    '''
    Executor generator for all stable diffusion API paths.
    '''
    batch_size = 4
    stable_diffusion_module = None

    def __init__(self,
        batch_size: int=4,
        height: int=512,
        max_n_subprompts=8,
        max_resolution=589824,
        n_iter: int=1,
        use_half: bool=False,
        weights_path='',
        width: int=512,
        **kwargs,
    ):
        '''
        @batch_size: The number of images to create at the same time. It only
          slightly speeds up inference while dramatically increasing memory
          usage.
        @height: Default height of image in pixels.
        @max_n_subprompts: Maximum number of subprompts you can add to an image
          in the denoising step. More subprompts = slower denoising.
        @max_resolution: The maximum resolution for images in pixels, to keep
          your GPU from OOMing in server applications.
        @n_iter: Default number of iterations for the sampler.
        @use_half: Sample with FP16 instead of FP32. Saves some memory for
          approximately the same results.
        @weights_path: Location of the Stable Diffusion weights checkpoint file.
        @width: Default width of image in pixels.
        '''
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.stable_diffusion_module = StableDiffusionInference(
            checkpoint_loc=weights_path,
            height=height,
            max_n_subprompts=max_n_subprompts,
            max_resolution=max_resolution,
            n_iter=n_iter,
            use_half=use_half,
            width=width,
        )
        
    def _h_and_w_from_parameters(self, parameters, opt):
        height = parameters.get('height', opt.height)
        if height is not None:
            height = int(height)
        else:
            height = opt.height
        width = parameters.get('width', opt.width)
        if width is not None:
            width = int(width)
        else:
            width = opt.width

        return height, width

    @requests(on='/')
    def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        request_time = time.time()

        # Default options for inherence engine.
        opt = self.stable_diffusion_module.opt

        sampler = parameters.get('sampler', 'k_lms')
        if sampler not in K_DIFF_SAMPLERS:
            raise ValueError(f'sampler must be in {K_DIFF_SAMPLERS}, got {sampler}')
        scale = parameters.get('scale', opt.scale)
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        steps = min(int(parameters.get('steps', opt.ddim_steps)), MAX_STEPS)
        height, width = self._h_and_w_from_parameters(parameters, opt)

        # If the number of samples we have is more than would currently be
        # given for batch_size * n_iter, increase n_iter to yield more images.
        n_samples = self.batch_size
        n_iter = opt.n_iter
        if num_images < n_samples:
            n_samples = num_images
        if num_images // n_samples > n_iter:
            n_iter = num_images // n_samples

        for d in docs:
            batch_size = n_samples
            prompt = d.text
            assert prompt is not None

            self.logger.info(f'stable diffusion start {num_images} images, prompt "{prompt}"...')
            for i in trange(n_iter, desc="Sampling"):
                samples, extra_data = self.stable_diffusion_module.sample(
                    prompt,
                    batch_size,
                    sampler,
                    seed + i,
                    steps,
                    height=height,
                    scale=scale,
                    width=width,
                )

                (
                    conditioning,
                    images,
                ) = itemgetter('conditioning', 'images')(extra_data)

                for img in images:
                    buffered = BytesIO()
                    img.save(buffered, format='PNG')

                    samples_buffer = BytesIO()
                    torch.save(samples, samples_buffer)
                    samples_buffer.seek(0)

                    _d = Document(
                        embedding=conditioning,
                        blob=buffered.getvalue(),
                        mime_type='image/png',
                        tags={
                            'latent_repr': base64.b64encode(
                                samples_buffer.getvalue()).decode(),
                            'request': {
                                'api': 'txt2img',
                                'height': height,
                                'num_images': num_images,
                                'sampler': sampler,
                                'scale': scale,
                                'seed': seed,
                                'steps': steps,
                                'width': width,
                            },
                            'text': prompt,
                            'generator': 'stable-diffusion',
                            'request_time': request_time,
                            'created_time': time.time(),
                        },
                    ).convert_blob_to_datauri()
                    _d.text = prompt
                    d.matches.append(_d)

                torch.cuda.empty_cache()

    @requests(on='/stablediffuse')
    def stablediffuse(self, docs: DocumentArray, parameters: Dict, **kwargs):
        '''
        Called "img2img" in the scripts of the stable-diffusion repo.
        '''
        request_time = time.time()

        # Default options for inherence engine.
        opt = self.stable_diffusion_module.opt

        latentless = parameters.get('latentless', False)
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        prompt_override = parameters.get('prompt', None)
        sampler = parameters.get('sampler', 'k_lms')
        scale = parameters.get('scale', opt.scale)
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        strength = parameters.get('strength', 0.75)

        if sampler not in K_DIFF_SAMPLERS:
            raise ValueError(f'sampler must be in {K_DIFF_SAMPLERS}, got {sampler}')

        steps = min(int(parameters.get('steps', opt.ddim_steps)), MAX_STEPS)

        # If the number of samples we have is more than would currently be
        # given for n_samples * n_iter, increase n_iter to yield more images.
        n_samples = self.batch_size
        n_iter = opt.n_iter
        if num_images < n_samples:
            n_samples = num_images
        if num_images // n_samples > n_iter:
            n_iter = num_images // n_samples

        assert 0. < strength < 1., 'can only work with strength in (0.0, 1.0)'

        for d in docs:
            batch_size = n_samples
            prompt = d.text
            if prompt_override is not None:
                prompt = prompt_override
            assert prompt is not None

            for i in trange(n_iter, desc="Sampling"):
                samples, extra_data = self.stable_diffusion_module.sample(
                    prompt,
                    batch_size,
                    sampler,
                    seed + i,
                    steps,
                    init_pil_image=document_to_pil(d),
                    init_pil_image_as_random_latent=latentless,
                    scale=scale,
                    strength=strength,
                )

                (
                    conditioning,
                    images,
                ) = itemgetter('conditioning', 'images')(extra_data)

                for img in images:
                    buffered = BytesIO()
                    img.save(buffered, format='PNG')

                    samples_buffer = BytesIO()
                    torch.save(samples, samples_buffer)
                    samples_buffer.seek(0)

                    _d = Document(
                        embedding=conditioning,
                        blob=buffered.getvalue(),
                        mime_type='image/png',
                        tags={
                            'latent_repr': base64.b64encode(
                                samples_buffer.getvalue()).decode(),
                            'request': {
                                'api': 'stablediffuse',
                                'latentless': latentless,
                                'num_images': num_images,
                                'sampler': sampler,
                                'scale': scale,
                                'seed': seed,
                                'steps': steps,
                                'strength': strength,
                            },
                            'text': prompt,
                            'generator': 'stable-diffusion',
                            'request_time': request_time,
                            'created_time': time.time(),
                        },
                    ).convert_blob_to_datauri()
                    _d.text = prompt
                    d.matches.append(_d)

                torch.cuda.empty_cache()

    @requests(on='/stableinterpolate')
    def stableinterpolate(self, docs: DocumentArray, parameters: Dict, **kwargs):
        '''
        Create a series of images that are interpolations between two prompts.
        '''
        request_time = time.time()

        # Default options for inherence engine.
        opt = self.stable_diffusion_module.opt

        num_images = max(1, min(16, int(parameters.get('num_images', 1))))
        resample_prior = parameters.get('resample_prior', True)
        sampler = parameters.get('sampler', 'k_lms')
        scale = parameters.get('scale', opt.scale)
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        strength = parameters.get('strength', 0.75)

        if sampler not in K_DIFF_SAMPLERS:
            raise ValueError(f'sampler must be in {K_DIFF_SAMPLERS}, got {sampler}')

        steps = min(int(parameters.get('steps', opt.ddim_steps)), MAX_STEPS)
        height, width = self._h_and_w_from_parameters(parameters, opt)

        assert 0.5 <= strength <= 1., 'can only work with strength in [0.5, 1.0]'

        for d in docs:
            batch_size = 1
            prompt = d.text
            assert prompt is not None

            prompts = prompt.split('|')

            (
                conditioning_start,
                unconditioning, # Reuse this as it's the same for both
                weighted_subprompts_start,
                _, # Don't need the individual embedding managers
            ) = self.stable_diffusion_module.compute_conditioning_and_weights(
                prompts[0].strip(),
                batch_size)

            (
                conditioning_end,
                _,
                weighted_subprompts_end,
                _, # Don't need the individual embedding managers
            ) = self.stable_diffusion_module.compute_conditioning_and_weights(
                prompts[1].strip(),
                batch_size)

            assert len(weighted_subprompts_start) == len(weighted_subprompts_end), \
                'Weighted subprompts for interpolation must be equal in number'

            to_iterate = list(enumerate(np.linspace(0, 1, num_images)))

            # Interate over interpolation percentages.
            samples_last = None
            for i, percent in to_iterate:
                c = None
                if i < 1:
                    c = conditioning_start
                elif i == len(to_iterate) - 1:
                    c = conditioning_end
                else:
                    c = conditioning_start.clone().detach()
                    for embedding_i, _ in enumerate(conditioning_start):
                        c[embedding_i] = slerp(
                            percent,
                            conditioning_start[embedding_i],
                            conditioning_end[embedding_i],
                        )
                weighted_subprompts = combine_weighted_subprompts(percent,
                    weighted_subprompts_start,
                    weighted_subprompts_end)

                image = None
                if i == 0 or not resample_prior:
                    samples_last, extra_data = self.stable_diffusion_module.sample(
                        prompt,
                        batch_size,
                        sampler,
                        seed + i,
                        steps,
                        conditioning=c,
                        height=height,
                        prompt_concept_injection_required=False,
                        scale=scale,
                        weighted_subprompts=weighted_subprompts,
                        width=width,
                        unconditioning=unconditioning,
                    )

                    (
                        image,
                    ) = itemgetter('images')(extra_data)
                else:
                    samples_last, extra_data = self.stable_diffusion_module.sample(
                        prompt,
                        batch_size,
                        sampler,
                        seed + i,
                        steps,
                        conditioning=c,
                        height=height,
                        init_latent=samples_last,
                        prompt_concept_injection_required=False,
                        scale=scale,
                        strength=strength,
                        weighted_subprompts=weighted_subprompts,
                        width=width,
                        unconditioning=unconditioning,
                    )

                    (
                        image,
                    ) = itemgetter('images')(extra_data)

                torch.cuda.empty_cache()

                buffered = BytesIO()
                image.save(buffered, format='PNG')

                samples_buffer = BytesIO()
                torch.save(samples_last, samples_buffer)
                samples_buffer.seek(0)

                _d = Document(
                    embedding=c,
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                    tags={
                        'latent_repr': base64.b64encode(
                            samples_buffer.getvalue()).decode(),
                        'request': {
                            'api': 'stableinterpolate',
                            'height': height,
                            'num_images': num_images,
                            'resample_prior': resample_prior,
                            'sampler': sampler,
                            'scale': scale,
                            'seed': seed,
                            'steps': steps,
                            'strength': strength,
                            'width': width,
                        },
                        'text': prompt,
                        'percent': percent,
                        'generator': 'stable-diffusion',
                        'request_time': request_time,
                        'created_time': time.time(),
                    },
                ).convert_blob_to_datauri()
                _d.text = prompt
                d.matches.append(_d)

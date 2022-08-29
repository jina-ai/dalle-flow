import PIL
import os
import shutil
import time
import torch

import k_diffusion as K
import numpy as np
import torch.nn as nn

from PIL import Image
from contextlib import nullcontext
from einops import rearrange, repeat
from io import BytesIO
from itertools import islice
from pathlib import Path
from pytorch_lightning import seed_everything
from random import randint
from torch import autocast
from tqdm import tqdm, trange
from typing import Dict

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from jina import Executor, DocumentArray, Document, requests
from omegaconf import OmegaConf

K_DIFF_SAMPLERS = {'k_lms', 'dpm2', 'dpm2_ancestral', 'heun',
    'euler', 'euler_ancestral'}
VALID_SAMPLERS = {'ddim', 'k_lms', 'dpm2', 'dpm2_ancestral', 'heun',
    'euler', 'euler_ancestral'}


class StableDiffusionConfig:
    '''
    Configuration for Stable Diffusion.
    '''
    C = 4 # latent channels
    ckpt = '' # model checkpoint path
    config = '' # model configuration file path
    ddim_eta = 0.0
    ddim_steps = 100
    f = 8 # downsampling factor
    fixed_code = False
    height = 512
    n_iter = 1 # number of times to sample
    n_samples = 1 # batch size, GPU memory use scales quadratically with this but it makes it sample faster!
    precision = 'autocast'
    scale = 7.5 # unconditional guidance scale
    seed = 1
    width = 512


class KCFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()
    return model


def load_img(path, img=None):
    image = None
    if img is None:
        image = Image.open(path).convert("RGB")
    else:
        image = img
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    helper function to spherically interpolate two arrays v1 v2

    from @xsteenbrugge
    """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


class StableDiffusionGenerator(Executor):
    '''
    Executor generator for all stable diffusion API paths.
    '''
    opt: StableDiffusionConfig = StableDiffusionConfig()

    config = ''
    device = None
    input_path = ''
    model = None
    model_k_wrapped = None
    model_k_config = None
    sampler = None

    def __init__(self,
        stable_path: str,
        height: int=512,
        n_iter: int=1,
        n_samples: int=4,
        width: int=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_path = stable_path
        self.opt.config = f'{stable_path}/configs/stable-diffusion/v1-inference.yaml'
        self.opt.ckpt = f'{stable_path}/models/ldm/stable-diffusion-v1/model.ckpt'

        self.opt.height = height
        self.opt.width = width
        self.opt.n_samples = n_samples
        self.opt.n_iter = n_iter

        self.config = OmegaConf.load(f"{self.opt.config}")
        self.model = load_model_from_config(self.config, f"{self.opt.ckpt}")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.model_k_wrapped = K.external.CompVisDenoiser(self.model)
        self.model_k_config = KCFGDenoiser(self.model_k_wrapped)

        self.sampler = DDIMSampler(self.model)

        self.sampler.make_schedule(
            ddim_num_steps=self.opt.ddim_steps, ddim_eta=self.opt.ddim_eta,
                verbose=False)

    def _sample_text(self, prompts, n_samples, batch_size, opt, sampler, steps,
        start_code):
        '''
        Create image(s) from text.
        '''
        uc = None
        if opt.scale != 1.0:
            uc = self.model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        c = self.model.get_learned_conditioning(prompts)
        shape = [opt.C, opt.height // opt.f, opt.width // opt.f]
        torch.cuda.empty_cache()

        samples = None
        if sampler == 'ddim':
            samples, _ = self.sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=start_code)
        if sampler in K_DIFF_SAMPLERS:
            # k_lms is the fallthrough
            sampling_fn = K.sampling.sample_lms
            if sampler == 'dpm2':
                sampling_fn = K.sampling.sample_dpm_2
            if sampler == 'dpm2_ancestral':
                sampling_fn = K.sampling.sample_dpm_2_ancestral
            if sampler == 'heun':
                sampling_fn = K.sampling.sample_heun
            if sampler == 'euler':
                sampling_fn = K.sampling.sample_euler
            if sampler == 'euler_ancestral':
                sampling_fn = K.sampling.sample_euler_ancestral

            sigmas = self.model_k_wrapped.get_sigmas(opt.ddim_steps)
            x = torch.randn([n_samples, *shape], device=self.device) * sigmas[0] # for GPU draw
            extra_args = {
                'cond': c,
                'uncond': uc,
                'cond_scale': opt.scale,
            }
            samples = sampling_fn(
                self.model_k_config,
                x,
                sigmas,
                extra_args=extra_args)

        x_samples_ddim = self.model.decode_first_stage(samples)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples_ddim

    @requests(on='/')
    def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        request_time = time.time()

        sampler = parameters.get('sampler', 'k_lms')
        if sampler not in VALID_SAMPLERS:
            raise ValueError(f'sampler must be in {VALID_SAMPLERS}, got {sampler}')
        scale = parameters.get('scale', 7.5)
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        opt = self.opt
        opt.scale = scale
        steps = int(parameters.get('steps', opt.ddim_steps))

        # If the number of samples we have is more than would currently be
        # given for n_samples * n_iter, increase n_iter to yield more images.
        n_samples = opt.n_samples
        n_iter = opt.n_iter
        if num_images < n_samples:
            n_samples = num_images
        if num_images // n_samples > n_iter:
            n_iter = num_images // n_samples
        seed_everything(seed)

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([n_samples, opt.C, opt.height // opt.f,
                opt.width // opt.f], device=self.device)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for d in docs:
                        batch_size = n_samples
                        prompt = d.text
                        assert prompt is not None
                        data = [batch_size * [prompt]]

                        self.logger.info(f'stable diffusion start {num_images} images, prompt "{prompt}"...')
                        for n in trange(n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data"):
                                x_samples_ddim = self._sample_text(prompts, n_samples,
                                    batch_size, opt, sampler, steps, start_code)
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    buffered = BytesIO()
                                    img.save(buffered, format='PNG')
                                    _d = Document(
                                        blob=buffered.getvalue(),
                                        mime_type='image/png',
                                        tags={
                                            'text': prompt,
                                            'generator': 'stable-diffusion',
                                            'request_time': request_time,
                                            'created_time': time.time(),
                                        },
                                    ).convert_blob_to_datauri()
                                    _d.text = prompt
                                    d.matches.append(_d)

    @requests(on='/stablediffuse')
    def stablediffuse(self, docs: DocumentArray, parameters: Dict, **kwargs):
        '''
        Called "img2img" in the scripts of the stable-diffusion repo.
        '''
        request_time = time.time()

        latentless = parameters.get('latentless', False)
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        prompt_override = parameters.get('prompt', None)
        sampler = parameters.get('sampler', 'k_lms')
        scale = parameters.get('scale', 7.5)
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        strength = parameters.get('strength', 0.75)

        if sampler not in VALID_SAMPLERS:
            raise ValueError(f'sampler must be in {VALID_SAMPLERS}, got {sampler}')

        opt = self.opt
        opt.scale = scale

        # If the number of samples we have is more than would currently be
        # given for n_samples * n_iter, increase n_iter to yield more images.
        n_samples = opt.n_samples
        n_iter = opt.n_iter
        if num_images < n_samples:
            n_samples = num_images
        if num_images // n_samples > n_iter:
            n_iter = num_images // n_samples

        seed_everything(seed)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * opt.ddim_steps)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for d in docs:
                        batch_size = n_samples
                        prompt = d.text
                        if prompt_override is not None:
                            prompt = prompt_override
                        assert prompt is not None
                        self.logger.info(f'stable diffusion img2img start {num_images} images, prompt "{prompt}"...')
                        data = [batch_size * [prompt]]

                        input_path = os.path.join(self.input_path, f'{d.id}/')

                        Path(input_path).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(input_path, 'out')).mkdir(parents=True, exist_ok=True)

                        temp_file_path = os.path.join(input_path, f'{d.id}.png')
                        d.save_uri_to_file(temp_file_path)

                        assert os.path.isfile(temp_file_path)
                        init_image = load_img(temp_file_path).to(self.device)
                        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

                        init_latent = None
                        if not latentless:
                            init_latent = self.model.get_first_stage_encoding(
                                self.model.encode_first_stage(init_image))  # move to latent space
                        else:
                            init_latent = torch.zeros(
                                batch_size,
                                4,
                                opt.height >> 3,
                                opt.width >> 3,
                            ).cuda()

                        for n in trange(n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if opt.scale != 1.0:
                                    uc = self.model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = self.model.get_learned_conditioning(prompts)

                                samples = None
                                if sampler == 'ddim':
                                    # encode (scaled latent)
                                    z_enc = self.sampler.stochastic_encode(
                                        init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                                    # decode it
                                    samples = self.sampler.decode(z_enc, c, t_enc,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc)
                                if sampler in K_DIFF_SAMPLERS:
                                    # k_lms is the fallthrough
                                    sampling_fn = K.sampling.sample_lms
                                    if sampler == 'dpm2':
                                        sampling_fn = K.sampling.sample_dpm_2
                                    if sampler == 'dpm2_ancestral':
                                        sampling_fn = K.sampling.sample_dpm_2_ancestral
                                    if sampler == 'heun':
                                        sampling_fn = K.sampling.sample_heun
                                    if sampler == 'euler':
                                        sampling_fn = K.sampling.sample_euler
                                    if sampler == 'euler_ancestral':
                                        sampling_fn = K.sampling.sample_euler_ancestral

                                    sigmas = self.model_k_wrapped.get_sigmas(opt.ddim_steps)
                                    x0 = init_latent
                                    noise = torch.randn_like(x0) * sigmas[opt.ddim_steps - t_enc - 1]
                                    xi = x0 + noise
                                    sigma_sched = sigmas[opt.ddim_steps - t_enc - 1:]
                                    extra_args = {
                                        'cond': c,
                                        'uncond': uc,
                                        'cond_scale': opt.scale,
                                    }
                                    samples = sampling_fn(
                                        self.model_k_config,
                                        xi,
                                        sigma_sched,
                                        extra_args=extra_args,
                                    )

                                x_samples = self.model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    buffered = BytesIO()
                                    img.save(buffered, format='PNG')
                                    _d = Document(
                                        blob=buffered.getvalue(),
                                        mime_type='image/png',
                                        tags={
                                            'text': prompt,
                                            'generator': 'stable-diffusion',
                                            'request_time': request_time,
                                            'created_time': time.time(),
                                        },
                                    ).convert_blob_to_datauri()
                                    _d.text = prompt
                                    d.matches.append(_d)

                            shutil.rmtree(input_path, ignore_errors=True)

    @requests(on='/stableinterpolate')
    def stableinterpolate(self, docs: DocumentArray, parameters: Dict, **kwargs):
        '''
        Create a series of images that are interpolations between two prompts.
        '''
        request_time = time.time()

        num_images = max(1, min(16, int(parameters.get('num_images', 1))))
        sampler = parameters.get('sampler', 'k_lms')
        scale = parameters.get('scale', 7.5)
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        strength = parameters.get('strength', 0.75)

        if sampler not in VALID_SAMPLERS:
            raise ValueError(f'sampler must be in {VALID_SAMPLERS}, got {sampler}')

        opt = self.opt
        opt.scale = scale

        seed_everything(seed)

        assert 0.5 <= strength <= 1., 'can only work with strength in [0.5, 1.0]'
        t_enc = int(strength * opt.ddim_steps)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for d in docs:
                        batch_size = 1
                        prompt = d.text
                        assert prompt is not None

                        prompts = prompt.split('|')
                        assert len(prompts) == 2, 'can only interpolate between two prompts'

                        self.logger.info(f'stable diffusion interpolate start {num_images} images, prompt "{prompt}"...')

                        prompt_embedding_start = self.model.get_learned_conditioning(prompts[0].strip())
                        prompt_embedding_end = self.model.get_learned_conditioning(prompts[1].strip())

                        to_iterate = list(enumerate(np.linspace(0, 1, num_images)))

                        # Interate over interpolation percentages.
                        last_image = None
                        x_samples = None
                        for i, percent in to_iterate:
                            init_image = None
                            init_latent = None
                            if i == 0:
                                start_code = None
                                if opt.fixed_code:
                                    start_code = torch.randn([1, opt.C, opt.height // opt.f,
                                        opt.width // opt.f], device=self.device)
                                x_samples = self._sample_text([prompts[0]], 1,
                                    batch_size, opt, sampler, opt.ddim_steps,
                                    start_code)
                            else:
                                init_image = load_img('', img=last_image).to(self.device)
                                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                                init_latent = self.model.get_first_stage_encoding(
                                    self.model.encode_first_stage(init_image))

                                uc = None
                                if opt.scale != 1.0:
                                    uc = self.model.get_learned_conditioning(batch_size * [""])

                                c = None
                                if i < 1:
                                    c = prompt_embedding_start
                                elif i == len(to_iterate) - 1:
                                    c = prompt_embedding_end
                                else:
                                    c = slerp(percent, prompt_embedding_start,
                                        prompt_embedding_end)

                                samples = None
                                if sampler == 'ddim':
                                    # encode (scaled latent)
                                    z_enc = self.sampler.stochastic_encode(
                                        init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                                    # decode it
                                    samples = self.sampler.decode(z_enc, c, t_enc,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc)
                                if sampler in K_DIFF_SAMPLERS:
                                    # k_lms is the fallthrough
                                    sampling_fn = K.sampling.sample_lms
                                    if sampler == 'dpm2':
                                        sampling_fn = K.sampling.sample_dpm_2
                                    if sampler == 'dpm2_ancestral':
                                        sampling_fn = K.sampling.sample_dpm_2_ancestral
                                    if sampler == 'heun':
                                        sampling_fn = K.sampling.sample_heun
                                    if sampler == 'euler':
                                        sampling_fn = K.sampling.sample_euler
                                    if sampler == 'euler_ancestral':
                                        sampling_fn = K.sampling.sample_euler_ancestral

                                sigmas = self.model_k_wrapped.get_sigmas(opt.ddim_steps)
                                x0 = init_latent
                                noise = torch.randn_like(x0) * sigmas[opt.ddim_steps - t_enc - 1]
                                xi = x0 + noise
                                sigma_sched = sigmas[opt.ddim_steps - t_enc - 1:]
                                extra_args = {
                                    'cond': c,
                                    'uncond': uc,
                                    'cond_scale': opt.scale,
                                }
                                samples = sampling_fn(
                                    self.model_k_config,
                                    xi,
                                    sigma_sched,
                                    extra_args=extra_args,
                                )

                                x_samples = self.model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))

                            buffered = BytesIO()
                            img.save(buffered, format='PNG')
                            last_image = img
                            _d = Document(
                                blob=buffered.getvalue(),
                                mime_type='image/png',
                                tags={
                                    'text': prompt,
                                    'percent': percent,
                                    'generator': 'stable-diffusion',
                                    'request_time': request_time,
                                    'created_time': time.time(),
                                },
                            ).convert_blob_to_datauri()
                            _d.text = prompt
                            d.matches.append(_d)

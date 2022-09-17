import os
import shutil
import sys
import time
import torch

from typing import Iterable, Optional

import k_diffusion as K
import numpy as np
import torch.nn as nn

from PIL import Image
from contextlib import nullcontext
from einops import rearrange, repeat
from io import BytesIO
from pathlib import Path
from pytorch_lightning import seed_everything
from random import randint
from torch import autocast
from tqdm import tqdm, trange
from typing import Dict

from ldm.models.diffusion.ddim import DDIMSampler

from jina import Executor, DocumentArray, Document, requests
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from util import (
    cat_self_with_repeat_interleaved,
    combine_weighted_subprompts,
    load_img,
    load_model_from_config,
    repeat_interleave_along_dim_0,
    slerp,
    split_weighted_subprompts_and_return_cond_latents,
    sum_along_slices_of_dim_0
)

K_DIFF_SAMPLERS = {'k_lms', 'dpm2', 'dpm2_ancestral', 'heun',
    'euler', 'euler_ancestral'}
VALID_SAMPLERS = {'ddim', 'k_lms', 'dpm2', 'dpm2_ancestral', 'heun',
    'euler', 'euler_ancestral'}


MAX_STEPS = 250
MIN_HEIGHT = 384
MIN_WIDTH = 384


class StableDiffusionConfig:
    '''
    Configuration for Stable Diffusion.
    '''
    C = 4 # latent channels
    ckpt = '' # model checkpoint path
    config = '' # model configuration file path
    ddim_eta = 0.0
    ddim_steps = 50
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

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        cond_scale: float,
        cond_arities: Iterable[int],
        cond_weights: Optional[Iterable[float]]
    ) -> torch.Tensor:
        '''
        Magicool k-sampler prompt positive/negative weighting from birch-san.

        https://github.com/Birch-san/stable-diffusion/blob/birch-mps-waifu/scripts/txt2img_fork.py
        '''
        uncond_count = uncond.size(dim=0)
        cond_count = cond.size(dim=0)
        cond_in = torch.cat((uncond, cond)).to(x.device)
        del uncond, cond
        cond_arities_tensor = torch.tensor(cond_arities, device=cond_in.device)
        # if x.dtype == torch.float32 or x.dtype == torch.float64:
        #     x = x.half()
        x_in = cat_self_with_repeat_interleaved(t=x,
            factors_tensor=cond_arities_tensor, factors=cond_arities,
            output_size=cond_count)
        del x
        sigma_in = cat_self_with_repeat_interleaved(t=sigma,
            factors_tensor=cond_arities_tensor, factors=cond_arities,
            output_size=cond_count)
        del sigma
        uncond_out, conds_out = self.inner_model(x_in, sigma_in, cond=cond_in) \
            .split([uncond_count, cond_count])
        del x_in, sigma_in, cond_in
        unconds = repeat_interleave_along_dim_0(t=uncond_out,
            factors_tensor=cond_arities_tensor, factors=cond_arities,
            output_size=cond_count)
        del cond_arities_tensor
        # transform
        #   tensor([0.5, 0.1])
        # into:
        #   tensor([[[[0.5000]]],
        #           [[[0.1000]]]])
        weight_tensor = torch.tensor(list(cond_weights),
            device=uncond_out.device, dtype=uncond_out.dtype) * cond_scale
        weight_tensor = weight_tensor.reshape(len(list(cond_weights)), 1, 1, 1)
        deltas: torch.Tensor = (conds_out-unconds) * weight_tensor
        del conds_out, unconds, weight_tensor
        cond = sum_along_slices_of_dim_0(deltas, arities=cond_arities)
        del deltas
        return uncond_out + cond


class StableDiffusionGenerator(Executor):
    '''
    Executor generator for all stable diffusion API paths.
    '''
    opt: StableDiffusionConfig = StableDiffusionConfig()

    config = ''
    device = None
    input_path = ''
    max_n_subprompts = None
    max_resolution = None
    model = None
    model_k_wrapped = None
    model_k_config = None
    sampler = None

    def __init__(self,
        stable_path: str,
        height: int=512,
        max_n_subprompts=8,
        max_resolution=589824,
        n_iter: int=1,
        n_samples: int=4,
        use_half: bool=False,
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

        self.max_n_subprompts = max_n_subprompts
        self.max_resolution = max_resolution

        self.config = OmegaConf.load(f"{self.opt.config}")
        self.model = load_model_from_config(self.config, f"{self.opt.ckpt}",
            use_half=use_half)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.model_k_wrapped = K.external.CompVisDenoiser(self.model)
        self.model_k_config = KCFGDenoiser(self.model_k_wrapped)

        self.sampler = DDIMSampler(self.model)

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

    def _height_and_width_check(self, height, width):
        if height * width > self.max_resolution:
            raise ValueError(f'height {height} and width {width} produce too ' +
                f'many pixels ({height * width}). Max pixels {self.max_resolution}')
        if height % 64 != 0:
            raise ValueError(f'height must be a multiple of 64 (got {height})')
        if width % 64 != 0:
            raise ValueError(f'width must be a multiple of 64 (got {width})')
        if height < MIN_HEIGHT:
            raise ValueError(f'width must be >= {MIN_HEIGHT} (got {height})')
        if width < MIN_WIDTH:
            raise ValueError(f'width must be >= {MIN_WIDTH} (got {width})')

    def _sample_text(self, prompt, n_samples, batch_size, opt, sampler, steps,
        start_code, c=None, weighted_subprompts=None, height=None, width=None):
        '''
        Create image(s) from text.
        '''

        self.sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=self.opt.ddim_eta,
                verbose=False)

        _height = opt.height if height is None else height
        _width = opt.width if width is None else width

        if isinstance(prompt, tuple) or isinstance(prompt, list):
            prompt = prompt[0]

        uc = self.model.get_learned_conditioning(batch_size * [""])
        if c is None:
            c, weighted_subprompts = split_weighted_subprompts_and_return_cond_latents(
                prompt,
                self.model.get_learned_conditioning,
                sampler,
                uc,
                max_n_subprompts=self.max_n_subprompts,
            )
        shape = [opt.C, _height // opt.f, _width // opt.f]

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

            sigmas = self.model_k_wrapped.get_sigmas(steps)
            x = torch.randn([n_samples, *shape], device=self.device) * sigmas[0] # for GPU draw
            extra_args = {
                'cond': c,
                'uncond': uc,
                'cond_scale': opt.scale,
                'cond_weights': [pr[1] for pr in weighted_subprompts] * batch_size,
                'cond_arities': (len(weighted_subprompts),) * batch_size,
            }
            samples = sampling_fn(
                self.model_k_config,
                x,
                sigmas,
                extra_args=extra_args)
        for i, _ in enumerate(samples):
            if samples[i].dtype == torch.float32 or samples[i].dtype == torch.float64:
                samples[i] = samples[i].half()
        x_samples_ddim = self.model.decode_first_stage(samples)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        torch.cuda.empty_cache()
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
        steps = min(int(parameters.get('steps', opt.ddim_steps)), MAX_STEPS)
        height, width = self._h_and_w_from_parameters(parameters, opt)
        self._height_and_width_check(height, width)

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
            start_code = torch.randn([n_samples, opt.C, height // opt.f,
                width // opt.f], device=self.device)

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
                                    batch_size, opt, sampler, steps, start_code,
                                    height=height, width=width)
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
        steps = min(int(parameters.get('steps', opt.ddim_steps)), MAX_STEPS)
        height, width = self._h_and_w_from_parameters(parameters, opt)
        self._height_and_width_check(height, width)

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
        t_enc = int(strength * steps)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    self.sampler.make_schedule(
                        ddim_num_steps=steps, ddim_eta=self.opt.ddim_eta,
                            verbose=False)

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
                                height >> 3,
                                width >> 3,
                            ).cuda()

                        for n in trange(n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data"):
                                uc = self.model.get_learned_conditioning(batch_size * [""])

                                c, weighted_subprompts = split_weighted_subprompts_and_return_cond_latents(
                                    prompt,
                                    self.model.get_learned_conditioning,
                                    sampler,
                                    uc,
                                    max_n_subprompts=self.max_n_subprompts,
                                )

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

                                    sigmas = self.model_k_wrapped.get_sigmas(steps)
                                    x0 = init_latent
                                    noise = torch.randn_like(x0) * sigmas[steps - t_enc - 1]
                                    xi = x0 + noise
                                    sigma_sched = sigmas[steps - t_enc - 1:]
                                    extra_args = {
                                        'cond': c,
                                        'uncond': uc,
                                        'cond_scale': opt.scale,
                                        'cond_weights': [pr[1] for pr in weighted_subprompts] * batch_size,
                                        'cond_arities': (len(weighted_subprompts),) * batch_size,
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
        resample_prior = parameters.get('resample_prior', True)
        sampler = parameters.get('sampler', 'k_lms')
        scale = parameters.get('scale', 7.5)
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        strength = parameters.get('strength', 0.75)

        if sampler not in VALID_SAMPLERS:
            raise ValueError(f'sampler must be in {VALID_SAMPLERS}, got {sampler}')

        opt = self.opt
        opt.scale = scale
        steps = min(int(parameters.get('steps', opt.ddim_steps)), MAX_STEPS)
        height, width = self._h_and_w_from_parameters(parameters, opt)
        self._height_and_width_check(height, width)

        seed_everything(seed)

        assert 0.5 <= strength <= 1., 'can only work with strength in [0.5, 1.0]'
        t_enc = int(strength * steps)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    self.sampler.make_schedule(
                        ddim_num_steps=steps, ddim_eta=self.opt.ddim_eta,
                            verbose=False)

                    for d in docs:
                        batch_size = 1
                        prompt = d.text
                        assert prompt is not None

                        prompts = prompt.split('|')
                        assert len(prompts) == 2, 'can only interpolate between two prompts'

                        self.logger.info(f'stable diffusion interpolate start {num_images} images, prompt "{prompt}"...')

                        uc = self.model.get_learned_conditioning(batch_size * [""])
                        prompt_embedding_start, weighted_subprompts_start = split_weighted_subprompts_and_return_cond_latents(
                            prompts[0].strip(),
                            self.model.get_learned_conditioning,
                            sampler,
                            uc,
                            max_n_subprompts=self.max_n_subprompts,
                        )
                        prompt_embedding_end, weighted_subprompts_end = split_weighted_subprompts_and_return_cond_latents(
                            prompts[1].strip(),
                            self.model.get_learned_conditioning,
                            sampler,
                            uc,
                            max_n_subprompts=self.max_n_subprompts,
                        )
                        assert len(weighted_subprompts_start) == len(weighted_subprompts_end), \
                            'Weighted subprompts for interpolation must be equal in number'

                        to_iterate = list(enumerate(np.linspace(0, 1, num_images)))

                        # Interate over interpolation percentages.
                        last_image = None
                        x_samples = None

                        for i, percent in to_iterate:
                            init_image = None
                            init_latent = None

                            c = None
                            if i < 1:
                                c = prompt_embedding_start
                            elif i == len(to_iterate) - 1:
                                c = prompt_embedding_end
                            else:
                                c = prompt_embedding_start.clone().detach()
                                for i, _ in enumerate(prompt_embedding_start):
                                    c[i] = slerp(percent, prompt_embedding_start[i],
                                        prompt_embedding_end[i])
                            weighted_subprompts = combine_weighted_subprompts(percent,
                                weighted_subprompts_start,
                                weighted_subprompts_end)

                            if i == 0 or not resample_prior:
                                start_code = None
                                if opt.fixed_code:
                                    start_code = torch.randn([1, opt.C, height // opt.f,
                                        width // opt.f], device=self.device)
                                x_samples = self._sample_text(None, 1,
                                    batch_size, opt, sampler, steps,
                                    start_code,
                                    c=c,
                                    height=height,
                                    width=width,
                                    weighted_subprompts=weighted_subprompts)
                            else:
                                init_image = load_img('', img=last_image).to(self.device)
                                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                                init_latent = self.model.get_first_stage_encoding(
                                    self.model.encode_first_stage(init_image))

                                uc = self.model.get_learned_conditioning(batch_size * [""])

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

                                    sigmas = self.model_k_wrapped.get_sigmas(steps)
                                    x0 = init_latent
                                    noise = torch.randn_like(x0) * sigmas[steps - t_enc - 1]
                                    xi = x0 + noise
                                    sigma_sched = sigmas[steps - t_enc - 1:]
                                    extra_args = {
                                        'cond': c,
                                        'uncond': uc,
                                        'cond_scale': opt.scale,
                                        'cond_weights': [pr[1] for pr in weighted_subprompts] * batch_size,
                                        'cond_arities': (len(weighted_subprompts),) * batch_size,
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

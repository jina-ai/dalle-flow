import re
from typing import Callable, Iterable

import PIL

import numpy as np
import torch

from PIL import Image
from itertools import islice
from ldm.util import instantiate_from_config


def cat_self_with_repeat_interleaved(
    t: torch.Tensor,
    factors: Iterable[int],
    factors_tensor: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Fast-paths for a pattern which in its worst-case looks like:
    t=torch.tensor([[0,1],[2,3]])
    factors=(2,3)
    torch.cat((t, t.repeat_interleave(factors, dim=0)))
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    Fast-path:
      `len(factors) == 1`
      it's just a normal repeat
    t=torch.tensor([[0,1]])
    factors=(2)
    tensor([[0, 1],
            [0, 1],
            [0, 1]])
    
    t=torch.tensor([[0,1],[2,3]])
    factors=(2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    """
    if len(factors) == 1:
        return repeat_along_dim_0(t, factors[0]+1)
    return torch.cat((t, repeat_interleave_along_dim_0(t=t,
        factors_tensor=factors_tensor, factors=factors, output_size=output_size))).to(t.device)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def combine_weighted_subprompts(alpha, wsp_a, wsp_b):
    wsps = []
    for i, _ in enumerate(wsp_a):
        wsps.append(tuple([
            wsp_a[i][0], # wsp_a[0] might not be the same? oh well
            wsp_b[i][1] * alpha + wsp_a[i][1] * (1 - alpha),
        ]))
    return wsps


def repeat_along_dim_0(t: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Repeats a tensor's contents along its 0th dim `factor` times.
    repeat_along_dim_0(torch.tensor([[0,1]]), 2)
    tensor([[0, 1],
            [0, 1]])
    # shape changes from (1, 2)
    #                 to (2, 2)
    
    repeat_along_dim_0(torch.tensor([[0,1],[2,3]]), 2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    # shape changes from (2, 2)
    #                 to (4, 2)
    """
    assert factor >= 1
    if factor == 1:
        return t
    if t.size(dim=0) == 1:
        # prefer expand() whenever we can, since doesn't copy
        return t.expand(factor * t.size(dim=0), *(-1,)*(t.ndim-1))
    return t.repeat((factor, *(1,)*(t.ndim-1)))


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


def load_model_from_config(config, ckpt, use_half=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()
    if use_half:
        model.half()
    return model


def repeat_interleave_along_dim_0(
    t: torch.Tensor,
    factors: Iterable[int],
    factors_tensor: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    repeat_interleave()s a tensor's contents along its 0th dim.
    factors=(2,3)
    factors_tensor = torch.tensor(factors)
    output_size=factors_tensor.sum().item() # 5
    t=torch.tensor([[0,1],[2,3]])
    repeat_interleave_along_dim_0(t=t, factors=factors, factors_tensor=factors_tensor, output_size=output_size)
    tensor([[0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    """
    factors_len = len(factors)
    assert factors_len >= 1
    if len(factors) == 1:
        # prefer repeat() whenever we can, because MPS doesn't support repeat_interleave()
        return repeat_along_dim_0(t, factors[0])
    if t.device.type != 'mps':
        return t.repeat_interleave(factors_tensor, dim=0, output_size=output_size)
    return torch.cat([repeat_along_dim_0(split, factor)
        for split, factor in zip(t.split(1, dim=0), factors)]).to(t.device)



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


def split_weighted_subprompts_and_return_cond_latents(
    text: str,
    get_learned_conditioning: Callable,
    sampler: str,
    unconditioned_prompt: torch.Tensor,
    max_n_subprompts: int=0,
    skip_normalize: bool=False,
):
    """
    Adapted from:
    https://github.com/lstein/stable-diffusion/
    https://github.com/lstein/stable-diffusion/blob/main/LICENSE

    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile("""
            (?P<prompt>     # capture group for 'prompt'
            (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
            )               # end 'prompt'
            (?:             # non-capture group
            :+              # match one or more ':' characters
            (?P<weight>     # capture group for 'weight'
            -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
            )?              # end weight capture group, make optional
            \s*             # strip spaces after weight
            |               # OR
            $               # else, if no ':' then match end of line
            )               # end non-capture group
            """, re.VERBOSE)
    parsed_prompts = [(match.group("prompt").replace("\\:", ":"), float(
        match.group("weight") or 1)) for match in re.finditer(prompt_parser, text)]
    weighted_subprompts = parsed_prompts

    c = None
    if len(weighted_subprompts) > 1 and sampler == 'ddim':
        # Use stupid weighting for DDIM sampler, since I don't feel like
        # hijacking the forward step. Negative weights won't work but it
        # doesn't matter because no one uses the ddim sampler.
        weight_sum = sum(map(lambda x: x[1], parsed_prompts))
        weighted_subprompts = [(x[0], x[1] / weight_sum) for x in parsed_prompts]

        c = torch.zeros_like(unconditioned_prompt)
        # normalize each "sub prompt" and add it
        for subprompt, weight in weighted_subprompts:
            c = torch.add(
                c,
                get_learned_conditioning([subprompt]),
                alpha=weight,
            )
    elif len(weighted_subprompts) > 1 and sampler != 'ddim':
        c = get_learned_conditioning([pr[0] for pr in weighted_subprompts])
    else:   # just standard 1 prompt
        c = get_learned_conditioning([text])

    if max_n_subprompts > 0 and len(weighted_subprompts) > max_n_subprompts:
        raise ValueError('The maximum allowed number of weighted subprompts ' +
            f'is {max_n_subprompts}, while {len(weighted_subprompts)} were ' +
            'generated')

    return c, weighted_subprompts


def sum_along_slices_of_dim_0(t: torch.Tensor, arities: Iterable[int]) -> torch.Tensor:
    """
    Implements fast-path for a pattern which in the worst-case looks like this:
    t=torch.tensor([[1],[2],[3]])
    arities=(2,1)
    torch.cat([torch.sum(split, dim=0, keepdim=True) for split in t.split(arities)])
    tensor([[3],
            [3]])
    Fast-path:
      `len(arities) == 1`
      it's just a normal sum(t, dim=0, keepdim=True)
    t=torch.tensor([[1],[2]])
    arities=(2)
    t.sum(dim=0, keepdim=True)
    tensor([[3]])
    """
    if len(arities) == 1:
        if t.size(dim=0) == 1:
            return t
        return t.sum(dim=0, keepdim=True)
    splits: List[Tensor] = t.split(arities)
    del t
    sums: List[Tensor] = [torch.sum(split, dim=0, keepdim=True) for split in splits]
    del splits
    return torch.cat(sums).to(t.device)
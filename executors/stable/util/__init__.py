import os
import re
import shutil
import urllib.request

from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List

import PIL

import numpy as np
import torch

from PIL import Image
from itertools import islice

from ldm.modules.encoders.modules import FrozenCLIPEmbedder, BERTEmbedder
from ldm.modules.embedding_manager import EmbeddingManager
from ldm.util import instantiate_from_config

TAGS_RE = re.compile('<.*?>')
sd_concepts_url_fn = lambda concept: f'https://huggingface.co/sd-concepts-library/{concept}/resolve/main/'
UNLIKELY_TOKENS = [
    '¯', '°', '±', '²', '³', '´', 'µ', '·', '¸', '¹', 'º', '»', '¼', '½', '¾',
]

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


def prompt_inject_custom_concepts(
    prompt: str,
    input_path: str,
    use_half: bool,
):
    '''
    Inject custom concepts from the sd-concepts-library into a prompt.
    '''
    def _next_token_for_concept():
        for token in UNLIKELY_TOKENS:
            yield token
        yield None

    def _get_clip_token_for_string(tokenizer, string):
        batch_encoding = tokenizer(string,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = batch_encoding['input_ids']

        if torch.count_nonzero(tokens - 49407) == 2:
            return tokens[0, 1]

        return None

    def _get_placeholder_loop(embedder):
        new_placeholder  = None
        token_seq = _next_token_for_concept()

        while True:
            new_placeholder = next(token_seq)
            if new_placeholder is None:
                raise ValueError('Ran out of tokens due to too many ' +
                    f'concepts (max: {len(UNLIKELY_TOKENS)})')

            token = _get_clip_token_for_string(
                embedder.tokenizer, new_placeholder)

            if token is not None:
                return new_placeholder, token

    prompt_injected = prompt

    # Check to see if we have a custom concept.
    embedding_manager = None
    embedding_folder = os.path.join(
        os.path.join(input_path, 'sd_embeddings_store'))
    Path(input_path).mkdir(parents=True, exist_ok=True)
    Path(embedding_folder).mkdir(parents=True, exist_ok=True)
    if '<' in prompt and '>' in prompt:
        embedding_paths = []
        for tag in re.findall(TAGS_RE, prompt):
            concept = tag[1:-1]

            # We found the concept, so dig it up and load it in. Store it
            # locally in case we need to use it again, too.
            Path(os.path.join(embedding_folder, concept)) \
                .mkdir(parents=True, exist_ok=True)

            tag_actual = None
            token_name_path = os.path.join(
                embedding_folder,
                f'{concept}/token_identifier.txt')
            concept_file_path = os.path.join(
                embedding_folder,
                f'{concept}/learned_embeds.bin')
            if not os.path.isfile(token_name_path):
                urllib.request.urlretrieve(
                    sd_concepts_url_fn(concept) + 'token_identifier.txt',
                    token_name_path)
                urllib.request.urlretrieve(
                    sd_concepts_url_fn(concept) + 'learned_embeds.bin',
                    concept_file_path)

            with open(token_name_path, 'r') as token_name_file:
                tag_actual = token_name_file.read()

            embedding_paths.append(concept_file_path)
            prompt_injected = prompt_injected.replace(tag, tag_actual)

        # Merge the embeddings.
        embedder = FrozenCLIPEmbedder().cuda()
        EmbeddingManagerCls = partial(EmbeddingManager, embedder, ["*"])

        string_to_token_dict = {}
        string_to_param_dict = torch.nn.ParameterDict()

        placeholder_to_src = {}

        for manager_ckpt in embedding_paths:
            manager = EmbeddingManagerCls()
            manager.load(manager_ckpt)
            if use_half:
                manager = manager.half()

            for placeholder_string in manager.string_to_token_dict:
                if not placeholder_string in string_to_token_dict:
                    string_to_token_dict[placeholder_string] = manager.string_to_token_dict[placeholder_string]
                    string_to_param_dict[placeholder_string] = manager.string_to_param_dict[placeholder_string]

                    placeholder_to_src[placeholder_string] = manager_ckpt
                else:
                    new_placeholder, new_token = _get_placeholder_loop(
                        embedder)
                    string_to_token_dict[new_placeholder] = new_token
                    string_to_param_dict[new_placeholder] = manager.string_to_param_dict[placeholder_string]

                    placeholder_to_src[new_placeholder] = manager_ckpt

        merged_manager = EmbeddingManagerCls()
        if use_half:
            manager = manager.half()
        merged_manager.string_to_param_dict = string_to_param_dict
        merged_manager.string_to_token_dict = string_to_token_dict
        embedding_manager = merged_manager

        shutil.rmtree(os.path.join(input_path, 'embeddings_tmp'), ignore_errors=True)
    return prompt_injected, embedding_manager

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
    embedding_manager,
    sampler: str,
    unconditioned_prompt: torch.Tensor,
    max_n_subprompts: int=0,
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
                get_learned_conditioning([subprompt], embedding_manager),
                alpha=weight,
            )
    elif len(weighted_subprompts) > 1 and sampler != 'ddim':
        c_learned = get_learned_conditioning([pr[0] for pr in weighted_subprompts],
            embedding_manager)
        c = torch.tile(c_learned, (unconditioned_prompt.size()[0], 1))
    else:   # just standard 1 prompt
        c_learned = get_learned_conditioning([text],
            embedding_manager)
        c = torch.tile(c_learned, (unconditioned_prompt.size()[0], 1))

    if max_n_subprompts > 0 and len(weighted_subprompts) > max_n_subprompts:
        raise ValueError('The maximum allowed number of weighted subprompts ' +
            f'is {max_n_subprompts}, while {len(weighted_subprompts)} were ' +
            'generated')

    if len(weighted_subprompts) == 0:
        weighted_subprompts = [('', 1.)]

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
    device = t.device
    del t
    sums: List[Tensor] = [torch.sum(split, dim=0, keepdim=True) for split in splits]
    del splits
    return torch.cat(sums).to(device)
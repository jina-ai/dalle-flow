'''
A simple python script that parses the flow.yml and removes any undesired
executors based on environmental variables that are present, then creates
flow*.tmp.yml.

Environmental flags available:

DISABLE_CLIP
DISABLE_DALLE_MEGA
DISABLE_GLID3XL
DISABLE_SWINIR
ENABLE_STABLE_DIFFUSION

TODO Support jcloud and k8s configurations?
'''
import argparse
import os
import sys
import yaml

from collections import OrderedDict

ENV_DISABLE_CLIP = 'DISABLE_CLIP'
ENV_DISABLE_DALLE_MEGA = 'DISABLE_DALLE_MEGA'
ENV_DISABLE_GLID3XL = 'DISABLE_GLID3XL'
ENV_DISABLE_SWINIR = 'DISABLE_SWINIR'
ENV_ENABLE_STABLE_DIFFUSION = 'ENABLE_STABLE_DIFFUSION'

ENV_GPUS_DALLE_MEGA = 'GPUS_DALLE_MEGA'
ENV_GPUS_GLID3XL = 'GPUS_GLID3XL'
ENV_GPUS_SWINIR = 'GPUS_SWINIR'
ENV_GPUS_STABLE_DIFFUSION = 'GPUS_STABLE_DIFFUSION'

FLOW_KEY_ENV = 'env'
FLOW_KEY_ENV_CUDA_DEV = 'CUDA_VISIBLE_DEVICES'

CAS_FLOW_NAME = 'clip_encoder'
DALLE_MEGA_FLOW_NAME = 'dalle'
GLID3XL_FLOW_NAME = 'diffusion'
RERANK_FLOW_NAME = 'rerank'
SWINIR_FLOW_NAME = 'upscaler'
STABLE_DIFFUSION_FLOW_NAME = 'stable'


def represent_ordereddict(dumper, data):
    '''
    Used to edit the YAML filters in place so that jina doesn't freak out when
    we use the newly parsed file. Otherwise the new YAML is sorted by keys and
    that breaks jina.
    '''
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

parser = argparse.ArgumentParser()

parser.add_argument('-fn','--filename',
    dest='filename',
    help='YAML file to use (default is flow.yaml)',
    required=False)
parser.add_argument('-o','--output',
    dest='output',
    help='YAML file to output (default is flow.tmp.yaml)',
    required=False)
parser.add_argument('--disable-clip',
    dest='no_clip',
    action='store_true',
    help="Disable clip-as-a-service executor (default false)",
    required=False)
parser.add_argument('--disable-dalle-mega',
    dest='no_dalle_mega',
    action='store_true',
    help="Disable DALLE-MEGA executor (default false)",
    required=False)
parser.add_argument('--disable-glid3xl',
    dest='no_glid3xl',
    action='store_true',
    help="Disable GLID3XL executor (default false)",
    required=False)
parser.add_argument('--disable-swinir',
    dest='no_swinir',
    action='store_true',
    help="Disable SWINIR upscaler executor (default false)",
    required=False)
parser.add_argument('--enable-stable-diffusion',
    dest='yes_stable_diffusion',
    action='store_true',
    help="Enable Stable Diffusion executor (default false)",
    required=False)
parser.add_argument('--gpus-dalle-mega',
    dest='gpus_dalle_mega',
    help="GPU device ID(s) for DALLE-MEGA (default 0)",
    default=0,
    required=False)
parser.add_argument('--gpus-glid3xl',
    dest='gpus_glid3xl',
    help="GPU device ID(s) for GLID3XL (default 0)",
    default=0,
    required=False)
parser.add_argument('--gpus-stable-diffusion',
    dest='gpus_stable_diffusion',
    help="GPU device ID(s) for Stable Diffusion (default 0)",
    default=0,
    required=False)
parser.add_argument('--gpus-swinir',
    dest='gpus_swinir',
    help="GPU device ID(s) for SWINIR (default 0)",
    default=0,
    required=False)

args = vars(parser.parse_args())

flow_to_use = 'flow.yml'
if args.get('filename', None) is not None:
    flow_to_use = args['filename']

output_flow = 'flow.tmp.yml'
if args.get('output', None) is not None:
    output_flow = args['output']

no_clip = args.get('no_clip') or \
    os.environ.get(ENV_DISABLE_CLIP, False)
no_dalle_mega = args.get('no_dalle_mega') or \
    os.environ.get(ENV_DISABLE_DALLE_MEGA, False)
no_glid3xl = args.get('no_glid3xl') or os.environ.get(ENV_DISABLE_GLID3XL, False)
no_swinir = args.get('no_swinir') or os.environ.get(ENV_DISABLE_SWINIR, False)
yes_stable_diffusion = args.get('yes_stable_diffusion') or \
    os.environ.get(ENV_ENABLE_STABLE_DIFFUSION, False)
gpus_dalle_mega = os.environ.get(ENV_GPUS_DALLE_MEGA, False) or \
    args.get('gpus_dalle_mega')
gpus_glid3xl = os.environ.get(ENV_GPUS_GLID3XL, False) or \
    args.get('gpus_glid3xl')
gpus_stable_diffusion = os.environ.get(ENV_GPUS_STABLE_DIFFUSION, False) or \
    args.get('gpus_stable_diffusion')
gpus_swinir = os.environ.get(ENV_GPUS_SWINIR, False) or \
    args.get('gpus_swinir')

STABLE_YAML_DICT = OrderedDict({
    'env': {
        'CUDA_VISIBLE_DEVICES': gpus_stable_diffusion,
        'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform',
    },
    'name': STABLE_DIFFUSION_FLOW_NAME,
    'replicas': 1,
    'timeout_ready': -1,
    'uses': f'executors/{STABLE_DIFFUSION_FLOW_NAME}/config.yml',
})

def _filter_out(flow_exec_list, name):
    return list(filter(lambda exc: exc['name'] != name, flow_exec_list))

with open(flow_to_use, 'r') as f_in:
    flow_as_dict = None
    try:
        flow_as_dict = OrderedDict(yaml.safe_load(f_in))
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)

    # For backwards compatibility, we inject the stable diffusion configuration
    # into the flow yml and then remove it if needed.
    #
    # Find the index of latent diffusion and inject stable diffusion after it.
    glid3xl_idx = next(i for i, exc in enumerate(flow_as_dict['executors'])
        if exc['name'] == GLID3XL_FLOW_NAME)
    flow_as_dict['executors'].insert(glid3xl_idx + 1, STABLE_YAML_DICT)

    # Find the rerank executor, jam stable into its needs.
    rerank_idx = next(i for i, exc in enumerate(flow_as_dict['executors'])
        if exc['name'] == RERANK_FLOW_NAME)
    flow_as_dict['executors'][rerank_idx]['needs'].append(
        STABLE_DIFFUSION_FLOW_NAME)

    if flow_as_dict is None:
        print('Input yaml was empty')
        sys.exit(1)

    if flow_as_dict.get('executors', None) is None:
        print('No executors found in yaml file')
        sys.exit(1)

    if no_dalle_mega:
        flow_as_dict['executors'] = _filter_out(flow_as_dict['executors'],
            DALLE_MEGA_FLOW_NAME)
    else:
        dalle_mega_idx = next(i for i, exc in enumerate(flow_as_dict['executors'])
            if exc['name'] == DALLE_MEGA_FLOW_NAME)
        flow_as_dict['executors'][dalle_mega_idx][FLOW_KEY_ENV][FLOW_KEY_ENV_CUDA_DEV] = gpus_dalle_mega

    if no_glid3xl:
        flow_as_dict['executors'] = _filter_out(flow_as_dict['executors'],
            GLID3XL_FLOW_NAME)
    else:
        glid3xl_idx = next(i for i, exc in enumerate(flow_as_dict['executors'])
            if exc['name'] == GLID3XL_FLOW_NAME)
        flow_as_dict['executors'][glid3xl_idx][FLOW_KEY_ENV][FLOW_KEY_ENV_CUDA_DEV] = gpus_glid3xl

    if no_swinir:
        flow_as_dict['executors'] = _filter_out(flow_as_dict['executors'],
            SWINIR_FLOW_NAME)
    else:
        swinir_idx = next(i for i, exc in enumerate(flow_as_dict['executors'])
            if exc['name'] == SWINIR_FLOW_NAME)
        flow_as_dict['executors'][swinir_idx][FLOW_KEY_ENV][FLOW_KEY_ENV_CUDA_DEV] = gpus_swinir

    if not yes_stable_diffusion:
        flow_as_dict['executors'] = _filter_out(flow_as_dict['executors'],
            STABLE_DIFFUSION_FLOW_NAME)

    for exc in flow_as_dict['executors']:
        if type(exc.get('needs', None)) == list:
            if no_dalle_mega:
                exc['needs'] = list(filter(
                    lambda _n: _n != DALLE_MEGA_FLOW_NAME,
                    exc['needs']))
            if no_glid3xl:
                exc['needs'] = list(filter(
                    lambda _n: _n != GLID3XL_FLOW_NAME,
                    exc['needs']))
            if no_swinir:
                exc['needs'] = list(filter(
                    lambda _n: _n != SWINIR_FLOW_NAME,
                    exc['needs']))
            if not yes_stable_diffusion:
                exc['needs'] = list(filter(
                    lambda _n: _n != STABLE_DIFFUSION_FLOW_NAME,
                    exc['needs']))

    if no_clip:
        flow_as_dict['executors'] = _filter_out(flow_as_dict['executors'],
            CAS_FLOW_NAME)
        flow_as_dict['executors'] = _filter_out(flow_as_dict['executors'],
            RERANK_FLOW_NAME)

    with open(output_flow, 'w') as f_out:
        f_out.write(yaml.dump(flow_as_dict))

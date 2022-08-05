import glob
import os
import shutil
import tempfile
import time
from typing import Dict
import json

from jina import Executor, DocumentArray, Document, requests


class GLID3Diffusion(Executor):
    def __init__(self, glid3_path: str, steps: int = 100, **kwargs):
        super().__init__(**kwargs)
        os.environ['GLID_MODEL_PATH'] = glid3_path
        os.environ['GLID3_STEPS'] = str(steps)
        self.diffusion_steps = steps
        from dalle_flow_glid3.model import static_args
        from dalle_flow_glid3.blank_encoding import generate_blank_embeddings

        assert static_args

        self.logger.info('Generating blank embeddings')
        with open(os.path.join(os.path.dirname(__file__), 'clip_blank_encoding.json')) as f:
            self.blank_bert_embedding, self.blank_clip_embedding = generate_blank_embeddings('a', json.load(f))

    def run_glid3(self, d: Document, text: str, skip_rate: float, num_images: int):
        request_time = time.time()

        with tempfile.NamedTemporaryFile(
                suffix='.png',
        ) as f_in:
            self.logger.info(f'diffusion [{text}] ...')
            from dalle_flow_glid3.cli_parser import parser

            kw = {
                'init_image': f_in.name if d.uri else None,
                'skip_timesteps': int(self.diffusion_steps * skip_rate) if d.uri else 0,
                'steps': self.diffusion_steps,
                'batch_size': num_images,
                'num_batches': 1,
                'text': f'"{text}"',
                'output_path': d.id
            }
            kw_str_list = []
            for k, v in kw.items():
                if v is not None:
                    kw_str_list.extend([f'--{k}', str(v)])
            if d.uri:
                d.save_uri_to_file(f_in.name)

            from dalle_flow_glid3.sample import do_run

            args = parser.parse_args(kw_str_list)
            do_run(args, d.embedding, self.blank_bert_embedding, self.blank_clip_embedding)

            kw.update({
                'generator': 'GLID3-XL',
                'request_time': request_time,
                'created_time': time.time(),
            })
            for f in glob.glob(f'{args.output_path}/*.png'):
                _d = Document(uri=f, text=d.text, tags=kw).convert_uri_to_datauri()
                d.matches.append(_d)

            # remove all outputs
            shutil.rmtree(args.output_path, ignore_errors=True)

            self.logger.info(f'done with [{text}]!')

    @requests
    def diffusion(self, docs: DocumentArray, parameters: Dict, **kwargs):
        skip_rate = float(parameters.get('skip_rate', 0.5))
        num_images = max(1, min(9, int(parameters.get('num_images', 1))))
        for d in docs:
            self.run_glid3(d, d.text, skip_rate=skip_rate, num_images=num_images)

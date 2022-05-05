import glob
import os
import shutil
import tempfile
from typing import Dict

from jina import Executor, DocumentArray, Document, requests


class GLID3Diffusion(Executor):
    def __init__(self, glid3_path: str, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.diffusion_steps = steps
        os.environ['GLID_MODEL_PATH'] = glid3_path
        from dalle_flow_glid3.sample import static_args
        self.default_args = static_args
        self.default_args.steps = self.diffusion_steps

        self.default_args.num_batches = 1

    def run_glid3(self, d: Document, text: str, skip_rate: float, num_images: int):
        with tempfile.NamedTemporaryFile(
                suffix='.png',
        ) as f_in:
            print(f'diffusion [{text}] ...')
            if d.uri:
                d.save_uri_to_file(f_in.name)
                self.default_args.init_image = f_in.name
            else:
                self.default_args.init_image = None
            print(self.default_args)
            self.default_args.skip_timesteps = int(self.diffusion_steps * skip_rate)
            self.default_args.text = text
            self.default_args.batch_size = num_images
            self.default_args.output_path = d.id

            from dalle_flow_glid3.sample import do_run

            do_run(self.default_args)

            for f in glob.glob(f'{self.default_args.output_path}/*.png'):
                _d = Document(uri=f, text=d.text).convert_uri_to_datauri()
                d.matches.append(_d)

            # remove all outputs
            shutil.rmtree(self.default_args.output_path, ignore_errors=True)

            print(f'done with [{text}]!')

    @requests(on='/')
    async def diffusion(self, docs: DocumentArray, parameters: Dict, **kwargs):
        skip_rate = float(parameters.get('skip_rate', 0.5))
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        for d in docs:
            self.run_glid3(d, d.text, skip_rate=skip_rate, num_images=num_images)

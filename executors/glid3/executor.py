import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from jina import Executor, DocumentArray, Document, requests


class GLID3Diffusion(Executor):
    def __init__(self, glid3_path: str, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.glid3_path = glid3_path
        self.diffusion_steps = steps

    def run_glid3(self, d: Document, text: str, skip_rate: float):
        os.chdir(self.glid3_path)
        with tempfile.NamedTemporaryFile(
            suffix='.png',
        ) as f_in:
            print(f'preparing {f_in.name} for diffusion...')
            d.save_uri_to_file(f_in.name)
            shutil.rmtree(f'{self.glid3_path}/output_npy', ignore_errors=True)

            Path(f'{self.glid3_path}/output').mkdir(parents=True, exist_ok=True)
            Path(f'{self.glid3_path}/output_npy').mkdir(parents=True, exist_ok=True)

            kw = {
                'init_image': f_in.name,
                'skip_timesteps': int(self.diffusion_steps * skip_rate),
                'steps': self.diffusion_steps,
                'model_path': 'finetune.pt',
                'batch_size': 4,
                'num_batches': 4,
                'text': f'"{text}"',
                'prefix': d.id,
            }
            kw_str = ' '.join(f'--{k} {str(v)}' for k, v in kw.items())
            print('diffusion...')
            print(subprocess.getoutput(f'python sample.py {kw_str}'))
            for f in glob.glob(f'{self.glid3_path}/output/{d.id}*.png'):
                kw['ctime'] = os.path.getctime(f)
                _d = Document(uri=f, tags=kw).convert_uri_to_datauri()
                d.matches.append(_d)

            # remove all outputs
            for f in glob.glob(f'{self.glid3_path}/output/{d.id}*.png'):
                if os.path.isfile(f):
                    os.remove(f)

            print('done!')

    @requests(on='/diffuse')
    async def diffusion(self, docs: DocumentArray, parameters: Dict, **kwargs):
        skip_rate = float(parameters.get('skip_rate', 0.5))
        for d in docs:
            self.run_glid3(d, d.text, skip_rate=skip_rate)

import glob
import os
import shutil
import subprocess
import tempfile

from jina import Executor, DocumentArray, Document, requests


class GLID3Diffusion(Executor):
    diffusion_steps = 80
    skip_rate = 0.5
    glid3_path = '/home/jupyter-han/glid-3-xl'
    top_k = 3

    def run_glid3(self, d: Document, text: str):
        os.chdir(self.glid3_path)
        with tempfile.NamedTemporaryFile(
            suffix='.png',
        ) as f_in:
            print(f'preparing {f_in.name}')
            d.save_blob_to_file(f_in.name)
            shutil.rmtree(f'{self.glid3_path}/output')
            os.mkdir(f'{self.glid3_path}/output')

            kw = {
                'init_image': f_in.name,
                'skip_timesteps': int(self.diffusion_steps * self.skip_rate),
                'steps': self.diffusion_steps,
                'model_path': 'finetune.pt',
                'batch_size': 6,
                'num_batches': 6,
                'text': f'"{text}"',
            }
            kw_str = ' '.join(f'--{k} {str(v)}' for k, v in kw.items())
            print('diffusion...')
            subprocess.getoutput(f'python sample.py {kw_str}')
            for f in glob.glob(
                f'{self.glid3_path}/output/*.png'
            ):
                kw['ctime'] = os.path.getctime(f)
                _d = Document(uri=f, tags=kw).load_uri_to_blob()
                d.matches.append(_d)
            print('done!')

    @requests
    async def diffusion(self, docs: DocumentArray, **kwargs):
        for d in docs:
            for m in d.matches[: self.top_k]:
                self.run_glid3(m, d.text)

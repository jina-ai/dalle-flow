import glob
import os
import subprocess
import tempfile

from jina import Executor, DocumentArray, Document, requests


class GLID3Diffusion(Executor):
    diffusion_steps = 80
    glid3_path = '/home/jupyter-han/glid-3-xl'
    top_k = 3

    def run_glid3(self, d: Document, text: str):
        os.chdir(self.glid3_path)
        with tempfile.NamedTemporaryFile(
            suffix='.png',
        ) as f_in:
            d.save_blob_to_file(f_in.name)

            kw = {
                'init_image': f_in.name,
                'skip_timesteps': 10,
                'steps': self.diffusion_steps,
                'model_path': 'finetune.pt',
                'batch_size': 6,
                'num_batches': 6,
                'text': f'"{text}"',
            }
            kw_str = ' '.join(f'--{k} {str(v)}' for k, v in kw.items())

            print(subprocess.getoutput(f'python sample.py {kw_str}'))
            list_of_files = glob.glob(
                f'{self.glid3_path}/output/*.png'
            )  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            diffu_c = Document(uri=latest_file, tags=kw)
            diffu_c.load_uri_to_blob()
            d.matches.append(diffu_c)

    @requests
    async def diffusion(self, docs: DocumentArray, **kwargs):
        for d in docs:
            for m in d.matches[: self.top_k]:
                self.run_glid3(m, d.text)

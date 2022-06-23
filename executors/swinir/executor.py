import glob
import os
import shutil
import subprocess
from pathlib import Path

from jina import Executor, DocumentArray, Document, requests


class SwinIRUpscaler(Executor):
    def __init__(self, swinir_path: str, **kwargs):
        super().__init__(**kwargs)
        self.swinir_path = swinir_path
        self.input_path = f'{swinir_path}/input/'
        self.output_path = f'{swinir_path}/results/swinir_real_sr_x4_large/'
        self.failover = 0

    def _upscale(self, d: Document):
        self.logger.info(f'upscaling [{d.text}]...')

        os.chdir(self.swinir_path)

        input_path = os.path.join(self.input_path, f'{d.id}/')

        Path(input_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        d.save_uri_to_file(os.path.join(input_path, f'{d.id}.png'))
        kw = {
            'task': 'real_sr',
            'scale': 4,
            'model_path': 'model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
            'folder_lq': input_path,
        }
        kw_str = ' '.join(f'--{k} {str(v)}' for k, v in kw.items())

        self.logger.info(
            subprocess.getoutput(f'python main_test_swinir.py --large_model {kw_str}')
        )
        d.uri = os.path.join(self.output_path, f'{d.id}_SwinIR.png')
        d.convert_uri_to_datauri()
        d.tags['upscaled'] = True
        d.tags.update(kw)

        self.logger.info('cleaning...')
        # remove input
        shutil.rmtree(input_path, ignore_errors=True)

        # remove all outputs
        for f in glob.glob(f'{self.output_path}/{d.id}*.png'):
            if os.path.isfile(f):
                os.remove(f)

        self.logger.info('done!')

    @requests(on='/upscale')
    async def upscale(self, docs: DocumentArray, **kwargs):
        for d in docs.find({'$and': [{'tags__upscaled': {'$exists': False}}, {'tags__generator': {'$exists': True}}]}):
            self._upscale(d)
            d.blob = None
            d.embedding = None

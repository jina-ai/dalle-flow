import time
import torch
from io import BytesIO
from typing import Dict
from min_dalle import MinDalle

from jina import Executor, requests, DocumentArray, Document

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MinDalle(
    models_root='dalle_pretrained',
    dtype=torch.float16,
    device=device,
    is_mega=True,
    is_reusable=True
)

class DalleGenerator(Executor):
    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):

        # can be of course larger but to save time and reduce the queue when serving public
        num_images = max(1, min(9, int(parameters.get('num_images', 1))))
        request_time = time.time()
        for d in docs:
            self.logger.info(f'dalle {num_images} [{str(d.text)[:100]}]...')

            for _ in range(num_images):
                img = model.generate_image(
                    text=d.text,
                    seed=-1,
                    grid_size=1,
                    is_seamless=False,
                    temperature=1,
                    top_k=256,
                    supercondition_factor=16,
                    is_verbose=False
                )

                buffered = BytesIO()
                img.save(buffered, format='PNG')
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                    tags={
                        'text': d.text,
                        'generator': 'DALLE-mega',
                        'request_time': request_time,
                        'created_time': time.time(),
                    },
                ).convert_blob_to_datauri()
                _d.text = d.text
                d.matches.append(_d)

            self.logger.info(f'done with [{str(d.text)[:100]}]...')

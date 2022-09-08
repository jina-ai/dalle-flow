import time
from io import BytesIO
from typing import Dict

from jina import Executor, requests, DocumentArray, Document

from . import dm_helper


class DalleGenerator(Executor):
    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):

        # can be of course larger but to save time and reduce the queue when serving public
        num_images = max(1, min(9, int(parameters.get('num_images', 1))))
        request_time = time.time()
        for d in docs:
            self.logger.info(f'dalle {num_images} [{d.text}]...')
            try:
                generated_imgs = dm_helper.generate_images(d.text, num_images)
            except RuntimeError as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'cudnn' in msg or 'resource_exhausted' in msg or 'xlaruntimeerror' in msg:
                    self.logger.error('| WARNING: ran out of memory, killing the process')
                    exit(1)
                else:
                    raise e

            for img in generated_imgs:
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

            self.logger.info(f'done with [{d.text}]')

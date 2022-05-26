from io import BytesIO
from typing import Dict

from jina import Executor, requests, DocumentArray, Document

import dm_helper


class DalleGenerator(Executor):

    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):

        # can be of course larger but to save time and reduce the queue when serving public
        num_images = max(1, min(9, int(parameters.get('num_images', 1))))
        for d in docs:
            generated_imgs = dm_helper.generate_images(d.text, num_images)

            for img in generated_imgs:
                buffered = BytesIO()
                img.save(buffered, format='PNG')
                _d = Document(blob=buffered.getvalue(), mime_type='image/png',
                              tags={'text': d.text, 'generator': 'DALLE-mega'}).convert_blob_to_datauri()
                _d.text = d.text
                d.matches.append(_d)


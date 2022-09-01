import time
from io import BytesIO
from typing import Dict

from jina import Executor, requests, DocumentArray, Document

from dalle_jina import *


class DalleGenerator(Executor):
    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Dalle()

    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):
        num_images = int(parameters.get('num_images', 1))
        for d in docs:
            generated_imgs = self.model.generate_images(d.text, num_images)

            for img in generated_imgs:
                _d = Document(
                    mime_type='image/png',
                    tags={  # TODO do we need these?
                        'text': d.text,
                        'generator': 'DALLE-mega',
                    },
                    tensor=img,
                ).convert_tensor_to_uri()
                _d.text = d.text  # TODO do we need this?
                d.matches.append(_d)

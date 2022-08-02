import os
import time
from io import BytesIO
from typing import Dict

from dalle2_laion import DalleModelManager, ModelLoadConfig, utils
from dalle2_laion.scripts import BasicInference

from jina import Executor, requests, DocumentArray, Document

# Defaults
#
# prior_batch_size: int = 100,
# decoder_batch_size: int = 10,
# prior_num_samples_per_batch: int = 2

DECODER_BATCH_SIZE = 36
PRIOR_BATCH_SIZE = 256
PRIOR_NUM_SAMPLES_PER_BATCH = 4
MODEL_PATH = '../dalle2-laion/configs/upsampler.example.json'

dreamer: BasicInference = BasicInference.create(MODEL_PATH, verbose=True)

class Dalle2Generator(Executor):
    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):
        # can be of course larger but to save time and reduce the queue when serving public
        num_images = max(1, min(9, int(parameters.get('num_images', 1))))
        request_time = time.time()
        for d in docs:
            self.logger.info(f'dalle2 {num_images} [{d.text}]...')
            output_map = dreamer.run([d.text],
                prior_batch_size=PRIOR_BATCH_SIZE,
                prior_num_samples_per_batch=PRIOR_NUM_SAMPLES_PER_BATCH,
                prior_sample_count=num_images,
                decoder_batch_size=DECODER_BATCH_SIZE,
            )

            for text in output_map:
                for embedding_index in output_map[text]:
                    for img in output_map[text][embedding_index]:
                        buffered = BytesIO()
                        img.save(buffered, format='PNG')
                        _d = Document(
                            blob=buffered.getvalue(),
                            mime_type='image/png',
                            tags={
                                'text': d.text,
                                'generator': 'DALLE2-1m',
                                'request_time': request_time,
                                'created_time': time.time(),
                            },
                        ).convert_blob_to_datauri()
                        _d.text = d.text
                        d.matches.append(_d)

            self.logger.info(f'done dalle2 with [{d.text}]')

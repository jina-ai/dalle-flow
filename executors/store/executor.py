import os
import time

os.environ['JINA_HUBBLE_REGISTRY'] = 'https://apihubble.staging.jina.ai'

from jina import Executor, requests, DocumentArray


class MyStore(Executor):

    @requests(on='/upscale')
    def store(self, docs: DocumentArray, **kwargs):
        docs[...].blobs = None  # remove all blobs from anywhere to save space
        docs[...].embeddings = None
        for d in docs.find({'$and': [{'tags__upscaled': {'$exists': True}}, {'tags__generator': {'$exists': True}}]}):
            d.tags['finish_time'] = time.time()
            DocumentArray([d]).push(f'dalle-flow-{d.id}')

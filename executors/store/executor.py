import time
import string
import random

from jina import Executor, requests, DocumentArray


class DalleFlowStore(Executor):

    @requests(on='/upscale')
    def store(self, docs: DocumentArray, **kwargs):
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        docs[...].blobs = None  # remove all blobs from anywhere to save space
        docs[...].embeddings = None
        for d in docs.find({'$and': [{'tags__upscaled': {'$exists': True}}, {'tags__generator': {'$exists': True}}]}):
            d.tags['finish_time'] = time.time()
            DocumentArray([d]).push(f'dalle-flow-{d.id}-{random_str}')

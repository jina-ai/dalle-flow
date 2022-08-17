from concurrent.futures import ThreadPoolExecutor
import os
import time

os.environ['JINA_HUBBLE_REGISTRY'] = 'https://apihubble.staging.jina.ai'

from jina import Executor, requests, DocumentArray, Document

class DalleFlowStore(Executor):

    def __init__(self, max_workers=4, **kwargs):
        super().__init__(**kwargs)
        # initializing ThreadPoolExecutor for pushing upscaled images
        self.__executor = ThreadPoolExecutor(max_workers=max_workers)

    def close(self):
        # closing ThreadPoolExecutor after all futures have finished
        self.__executor.shutdown()

    @requests(on='/upscale')
    def store(self, docs: DocumentArray, **kwargs):
        docs[...].blobs = None  # remove all blobs from anywhere to save space
        docs[...].embeddings = None
        for d in docs.find({'$and': [{'tags__upscaled': {'$exists': True}}, {'tags__generator': {'$exists': True}}]}):
            d.tags['finish_time'] = time.time()
            # submiting task to ThreadPoolExecutor
            self.__executor.submit(self.__store_doc, d)

    @staticmethod
    def __store_doc(doc: Document):
        DocumentArray([d]).push(f'dalle-flow-{d.id}')

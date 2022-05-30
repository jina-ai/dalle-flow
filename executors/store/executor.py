import os

os.environ['JINA_HUBBLE_REGISTRY'] = 'https://apihubble.staging.jina.ai'

from jina import Executor, requests, DocumentArray


class MyStore(Executor):

    def __init__(self, store_path: str, **kwargs):
        super().__init__(**kwargs)
        self.storage = DocumentArray(storage='sqlite', config={'connection': store_path, 'table_name': f'dallemega'})

    @requests(on='/upscale')
    def store(self, docs: DocumentArray, **kwargs):
        self.logger.info(f'store: handling request {docs[0].tags["request"]}')
        docs[...].blobs = None  # remove all blobs from anywhere to save space
        docs[...].embeddings = None
        for d in docs.find({'tags__upscaled': {'$exists': True}}):
            if d.id not in self.storage:
                self.storage.append(d)
                DocumentArray([d]).push(f'dalle-flow-{d.id}')
                self.logger.info(f'total: {len(self.storage)}')
        self.logger.info(f'store: finished request {docs[0].tags["request"]}')

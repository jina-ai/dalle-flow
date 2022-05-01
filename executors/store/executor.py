import os.path

from jina import Executor, requests, DocumentArray


class MyStore(Executor):

    def __init__(self, store_path: str, **kwargs):
        super().__init__(**kwargs)
        self.store_path = store_path
        if os.path.exists(self.store_path):
            self.storage = DocumentArray.load_binary(self.store_path)
        else:
            self.storage = DocumentArray()

    @requests
    def store(self, docs: DocumentArray, **kwargs):
        self.storage.extend(docs)
        self.storage.save_binary(self.store_path)
        print(f'total: {len(self.storage)}')

    @requests(on='/showall')
    def showall(self, docs: DocumentArray, **kwargs):
        return self.storage

    @requests(on='/upload')
    def upload(self, docs: DocumentArray, **kwargs):
        self.storage.push('dalle')

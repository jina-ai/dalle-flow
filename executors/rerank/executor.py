from clip_client import Client
from jina import Executor, requests, DocumentArray


class ReRank(Executor):

    def __init__(self, clip_server: str, **kwargs):
        super().__init__(**kwargs)
        self._client = Client(server=clip_server)

    @requests(on='/')
    async def rerank(self, docs: DocumentArray, **kwargs):
        return await self._client.arank(docs)

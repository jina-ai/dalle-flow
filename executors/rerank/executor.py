from clip_client import Client
from jina import Executor, requests, DocumentArray


class ReRank(Executor):

    def __init__(self, clip_server: str, **kwargs):
        super().__init__(**kwargs)
        self._client = Client(server=clip_server)

    @requests(on='/')
    async def rerank(self, docs: DocumentArray, **kwargs):
        print('re-rank, requests:', {d.tags['request'] for d in docs})
        for d in docs:
            print('re-rank, received request:', d.tags['request'], 'received matches datauris:', '\n'.join([
                m.uri for m in d.matches
            ]))

        print(docs.texts)
        docs = await self._client.arank(docs)
        print(docs.texts)
        for d in docs:
            print('re-rank, processed request (after clip):', d.tags['request'], 'received matches datauris:', '\n'.join([
                m.uri for m in d.matches
            ]))
        return docs

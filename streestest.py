# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from docarray import Document, DocumentArray
from multiprocessing import Process, Queue
import time
import uuid

str(uuid.uuid1())

server_url = 'grpcs://dalle-flow.dev.jina.ai'

prompt = 'a realistic photo of cat doing stand up paddle'


def post_request(queue):
    """Full dalle flow run"""
    id_ = str(uuid.uuid1())
    init_time= time.time()
    
    print(f'hello {id_} \n')
    
    da = Document(text=prompt).post(server_url, parameters={'num_images': 2}).matches
    fav_id = 0
    fav = da[fav_id]
    diffused = fav.post(f'{server_url}', parameters={'skip_rate': 0.6, 'num_images': 16}, target_executor='diffusion').matches
    dfav_id = 3
    fav = diffused[dfav_id]
    fav = fav.post(f'{server_url}/upscale')
    
    print(f'byebye {id_} \n time spent: {time.time()-init_time} \n queue size: {q.qsize()}')
    print(time.time()-init_time)
    q.put(id_)


N = 15

q = Queue()

list_thread = [Process(target=post_request, args=(q,), daemon=True) for _ in range(N)]

# + tags=[]
# %%time
for t in list_thread:
    t.start()
for t in list_thread:
    t.join()
# -

print(f'{N - q.qsize() } failed')

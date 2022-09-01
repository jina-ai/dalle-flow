#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from jina import Document

def call_server(prompt, idx):
    times = []
    try:
        timer = time.perf_counter()
        server_url = 'grpcs://621f4d6820.wolf.jina.ai'  

        print(f'sending request for #:{idx}, prompt: {prompt}\n')
        doc = Document(text=f'{idx} {prompt}').post(server_url, parameters={'num_images': 4})
        da = doc.matches
        # da.plot_image_sprites(fig_size=(10,10), show_index=True)
        print(f'==== 1st step #:{idx} -> first step finished in {(time.perf_counter() - timer):.2f}s')
        #raise Exception('stop here')
        times.append(time.perf_counter() - timer)
        timer = time.perf_counter()

        fav_id = 0

        fav = da[fav_id]
        fav.embedding = doc.embedding

        # fav.display()

        diffused = fav.post(f'{server_url}', parameters={'skip_rate': 0.6, 'num_images': 4}, 
                            target_executor='diffusion').matches
        # diffused.plot_image_sprites(fig_size=(10,10), show_index=True)
        print(f'-- 2nd step #:{idx} -> second step finished in {(time.perf_counter() - timer):.2f}s')
        times.append(time.perf_counter() - timer)
        timer = time.perf_counter()

        dfav_id = 0

        fav = diffused[dfav_id]

        # fav.display()

        fav = fav.post(f'{server_url}/upscale')
        # fav.display()
        times.append(time.perf_counter() - timer)

        assert fav.uri
        print(f'#:{idx} -> finished request. 3rd step took {(time.perf_counter() - timer):.2f}s')
        return times
    except Exception as err:
        print(f'#:{idx} error -> {err}')
        return times + [-1] * (3-len(times))

prompt_samples = [
    'an oil painting of a humanoid robot playing chess in the style of Matisse',
    'red and blue lightning striking at city during the night',
    'sunset with green landscape by the river',
    'playing football with huge crowd of supports'
]

n_requests = 50

prompts = [random.choice(prompt_samples) for _ in range(n_requests)]
with ProcessPoolExecutor(max_workers=n_requests) as executor:
    times = list(executor.map(call_server, prompts, range(n_requests)))

df = pd.DataFrame([
  {
      'gen_images': element[0],
      'diffusion': element[1],
      'upscale': element[2],
  }
  for element in times
])
df['total'] = df['gen_images'] + df['diffusion'] + df['upscale']
print(df)

df.describe()

print('all requests', df.shape)
print('failed requests', df[df['total'] < 0].shape)

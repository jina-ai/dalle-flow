import os
import subprocess
import tempfile
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from jina import Executor, requests, DocumentArray, Document


def _upscale(waifu_path: str, d: Document):
    f_in = tempfile.NamedTemporaryFile(
        'w',
        suffix='.png',
        delete=False,
    ).name
    f_out = tempfile.NamedTemporaryFile(
        'w',
        suffix='.png',
        delete=False,
    ).name

    d.save_blob_to_file(f_in)
    print(subprocess.getoutput(
        f'{waifu_path} -i {f_in} -o {f_out} -s 4 -n 0'))
    d.uri = f_out
    d.convert_uri_to_datauri()
    d.save_uri_to_file(f'dalle/{d.id}.png')
    return d


class Upscaler(Executor):

    def __init__(self,
                 waifu_url: str,
                 **kwargs):
        super().__init__(**kwargs)
        print('downloading...')
        resp = urlopen(waifu_url)
        zipfile = ZipFile(BytesIO(resp.read()))
        bin_path = './waifu-bin'
        zipfile.extractall(bin_path)
        print('complete')
        self.waifu_path = os.path.realpath(f'{bin_path}/waifu2x-ncnn-vulkan')
        print(self.waifu_path)

    @requests
    async def upscale(self, docs: DocumentArray, **kwargs):
        docs.apply(lambda d: _upscale(self.waifu_path, d))

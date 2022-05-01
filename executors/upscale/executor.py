import os
import stat
import subprocess
import tempfile
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from jina import Executor, requests, DocumentArray, Document


class Upscaler(Executor):
    def __init__(self, waifu_url: str, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        print('downloading...')
        resp = urlopen(waifu_url)
        zipfile = ZipFile(BytesIO(resp.read()))
        bin_path = './waifu-bin'
        zipfile.extractall(bin_path)
        print('complete')
        self.waifu_path = os.path.realpath(
            f'{bin_path}/waifu2x-ncnn-vulkan-20220419-ubuntu/waifu2x-ncnn-vulkan'
        )
        self.top_k = top_k

        st = os.stat(self.waifu_path)
        os.chmod(self.waifu_path, st.st_mode | stat.S_IEXEC)
        print(self.waifu_path)

    def _upscale(self, d: Document):
        with tempfile.NamedTemporaryFile(
                suffix='.png',
        ) as f_in, tempfile.NamedTemporaryFile(
            suffix='.png',
        ) as f_out:
            d.save_blob_to_file(f_in.name)
            print(
                subprocess.getoutput(
                    f'{self.waifu_path} -i {f_in.name} -o {f_out.name} -s 4 -n 0 -g -1'
                )
            )
            print(f'{f_in.name} done')
            d.uri = f_out.name
            d.convert_uri_to_datauri()
            d.blob = None
            d.tags['upscaled'] = 'true'
        return d

    @requests(on='/upscale')
    async def upscale(self, docs: DocumentArray, **kwargs):
        docs.apply(self._upscale)

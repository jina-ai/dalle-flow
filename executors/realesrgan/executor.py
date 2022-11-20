import enum
import time

from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen

import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from jina import Executor, DocumentArray, Document, requests
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


GFPGAN_MODEL_NAME = "GFPGANv1.3.pth"
GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
)


class RESRGAN_MODELS(str, enum.Enum):
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRNet_x4plus = "RealESRNet_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"
    RealESRGAN_x2plus = "RealESRGAN_x2plus"
    RealESR_animevideov3 = "realesr-animevideov3"
    RealESR_general_x4v3 = "realesr-general-x4v3"


class RealESRGANUpscaler(Executor):
    """
    This is a module that provides access to the RealESRGAN models and API which
    upscale images and video. It also supports using GFPGAN to fix faces within
    photographic images.

    The module source code is available at:
    https://github.com/xinntao/Real-ESRGAN

    All models that are included in the config.yml file will be available for
    upscaling.
    """
    cache_path: str | Path = ''
    gfpgan_weights_path: str | Path = ''
    models_to_load: List[str] = []
    pre_pad = 10
    tile = 0
    tile_pad = 10
    use_half = True

    def __init__(
        self,
        cache_path: str | Path,
        models_to_load: List[str],
        pre_pad: int = 10,
        tile: int = 0,
        tile_pad: int = 10,
        use_half: bool = True,
        **kwargs,
    ):
        """
        Args:

        cache_path: path to the cache directory.
        models_to_load: list[str], list of the models to load into memory.

        tile (int): As too large images result in the out of GPU memory issue,
          so this tile option will first crop input images into tiles, and
          then process each of them. Finally, they will be merged into one
          image.
          0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts.
          Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts.
          Default: 10.
        half (float): Whether to use half precision during inference.
          Default: True.
        """
        super().__init__(**kwargs)
        if "~" in str(Path(cache_path)):
            cache_path = Path(cache_path).expanduser()

        # Download/find weights for GFPGAN.
        gfpgan_weights_path = Path.home() / str(GFPGAN_MODEL_NAME + ".pth")
        if Path(cache_path).is_dir():
            gfpgan_weights_path = Path(cache_path) / str(GFPGAN_MODEL_NAME + ".pth")

        if not gfpgan_weights_path.is_file():
            # Assume we're working locally, use local home.
            gfpgan_weights_dir = Path.home()
            gfpgan_weights_path = Path.home() / str(GFPGAN_MODEL_NAME + ".pth")
            # weights_path will be updated
            gfpgan_weights_path = load_file_from_url(
                url=GFPGAN_MODEL_URL,
                model_dir=str(gfpgan_weights_dir.absolute()),
                progress=True,
                file_name=None,
            )

        self.cache_path = cache_path
        self.gfpgan_weights_path = gfpgan_weights_path
        self.models_to_load = models_to_load
        self.pre_pad = pre_pad
        self.tile = tile
        self.tile_pad = tile_pad
        self.use_half = use_half

    def load_model(self) -> Dict[str, Any]:
        '''
        return a dictionary organized as:
        {
        model_name: {
            'name': str,
            'netscale': int, (scaling strength eg 4=4x)
            'model': initialized RealESRGAN model,
            'model_face_fix': initialized GFPGAN model, [optional, non-anime only]
        }
        '''
        def gfpgan_wrapper(model_upscaler: Any, outscale: int):
            return GFPGANer(
                model_path=str(self.gfpgan_weights_path.absolute())
                if isinstance(self.gfpgan_weights_path, Path)
                else self.gfpgan_weights_path,
                upscale=outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=model_upscaler,
            )

        resrgan_models: Dict[str, Any] = {}
        for model_name in self.models_to_load:
            model_type = None
            try:
                model_type = RESRGAN_MODELS(model_name)
            except ValueError:
                raise ValueError(
                    f"Unknown model name '{model_name}', "
                    + "please ensure all models in models_to_load configuration "
                    + "option are valid"
                )

            model = None
            netscale = 4
            file_url = []
            if model_type == RESRGAN_MODELS.RealESRGAN_x4plus:  # x4 RRDBNet model
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
                file_url = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                ]
            if model_type == RESRGAN_MODELS.RealESRNet_x4plus:  # x4 RRDBNet model
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
                file_url = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
                ]
            if (
                model_type == RESRGAN_MODELS.RealESRGAN_x4plus_anime_6B
            ):  # x4 RRDBNet model with 6 blocks
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
                file_url = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
                ]
            if model_type == RESRGAN_MODELS.RealESRGAN_x2plus:  # x2 RRDBNet model
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                netscale = 2
                file_url = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                ]
            if (
                model_type == RESRGAN_MODELS.RealESR_animevideov3
            ):  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=16,
                    upscale=4,
                    act_type="prelu",
                )
                netscale = 4
                file_url = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
                ]
            if (
                model_type == RESRGAN_MODELS.RealESR_general_x4v3
            ):  # x4 VGG-style model (S size)
                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=32,
                    upscale=4,
                    act_type="prelu",
                )
                netscale = 4
                file_url = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                ]

            # determine model paths
            weights_path = Path.home() / str(model_name + ".pth")
            if Path(self.cache_path).is_dir():
                weights_path = Path(self.cache_path) / str(model_name + ".pth")

            if not weights_path.is_file():
                # Assume we're working locally, use local home.
                weights_dir = Path.home()
                weights_path = Path.home() / str(model_name + ".pth")
                for url in file_url:
                    # weights_path will be updated
                    weights_path = load_file_from_url(
                        url=url,
                        model_dir=str(weights_dir.absolute()),
                        progress=True,
                        file_name=None,
                    )

            # restorer
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(weights_path.absolute())
                if isinstance(weights_path, Path)
                else weights_path,
                model=model,
                tile=self.tile,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.use_half,
            )

            model_face_fix = None
            if model_type != RESRGAN_MODELS.RealESRGAN_x4plus_anime_6B:
                model_face_fix = gfpgan_wrapper(upsampler, netscale)

            resrgan_models[model_name] = {
                "name": model_name,
                "netscale": netscale,
                "model": upsampler,
                "model_face_fix": model_face_fix,
            }

        return resrgan_models

    def document_to_pil(self, doc):
        uri_data = urlopen(doc.uri)
        return Image.open(BytesIO(uri_data.read()))

    @requests(on="/realesrgan")
    def realesrgan(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Upscale using RealESRGAN, with or without face fix.

        @parameters.face_enhance: Whether or not to attempt to fix a human face.
          Not applicable to anime model. bool.
        @parameters.model_name: Which model to use, see RESRGAN_MODELS enum.
          str.
        """
        request_time = time.time()
        resrgan_models = self.load_model()

        face_enhance = parameters.get("face_enhance", False)
        model_name = parameters.get(
            "model_name", list(resrgan_models.values())[0]["name"]
        )

        for doc in docs:
            img = self.document_to_pil(doc)
            img_arr = np.asarray(img)

            model_dict = resrgan_models.get(model_name, None)
            if model_dict is None:
                raise ValueError(f"Unknown RealESRGAN upscaler specified: {model_name}")
            upsampler = model_dict.get("model", None)
            face_enhancer = model_dict.get("model_face_fix", None)
            if face_enhance is True and face_enhancer is not None:
                _, _, output = face_enhancer.enhance(
                    img_arr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
            else:
                output, _ = upsampler.enhance(img_arr, model_dict["netscale"])
            image_big = Image.fromarray(output)

            buffered = BytesIO()
            image_big.save(buffered, format="PNG")
            _d = Document(
                blob=buffered.getvalue(),
                mime_type="image/png",
                tags={
                    "request": {
                        "api": "realesrgan",
                        "face_enhance": face_enhance,
                        "model_name": model_name,
                    },
                    "text": doc.text,
                    "generator": "realesrgan",
                    "request_time": request_time,
                    "created_time": time.time(),
                },
            ).convert_blob_to_datauri()
            _d.text = doc.text
            doc.matches.append(_d)

        torch.cuda.empty_cache()

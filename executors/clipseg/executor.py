import enum
import os
import sys
import shutil
import time

from PIL import Image, ImageOps
from io import BytesIO
from pathlib import Path
from typing import Dict
from urllib.request import urlopen

import cv2
import numpy as np
import torch

from models.clipseg import CLIPDensePredT
from jina import Executor, DocumentArray, Document, requests
from torchvision import transforms


class THRESHOLDING_METHODS(str, enum.Enum):
    NONE = 'none' # Do not threshold
    BINARY = 'binary'
    ADAPTIVE_MEAN = 'adaptive_mean'
    ADAPTIVE_GAUSSIAN = 'adaptive_gaussian'


THRESHOLD_ADAPTIVE_DEFAULT_BLOCK_SIZE = 11
THRESHOLD_ADAPTIVE_DEFAULT_C = 2.
THRESHOLD_BINARY_DEFAULT_STRENGTH_VALUE = 85

WEIGHT_FOLDER_NAME = 'clipseg_weights'
WEIGHT_URL_DEFAULT = 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download'
WEIGHT_ZIP_FILE_NAME = 'clipseg_weights.zip'

class ClipSegmentation(Executor):
    model = None
    transformation = None

    def __init__(self,
        cache_path: str|Path,
        weights_url: str=WEIGHT_URL_DEFAULT,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if '~' in str(Path(cache_path)):
            cache_path = Path(cache_path).expanduser()

        weights_path = Path('/')
        if Path(cache_path).is_dir():
            weights_path = Path(cache_path) / WEIGHT_ZIP_FILE_NAME
        else:
            # Assume we're working locally, use local home.
            weights_path = Path.home() / WEIGHT_ZIP_FILE_NAME

        if not weights_path.is_file():
            response = urlopen(weights_url)
            weights_bytes = response.read()
            with open(weights_path, 'wb') as w_f:
                w_f.write(weights_bytes)

        shutil.unpack_archive(weights_path, Path(cache_path).resolve())

        model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
        model.eval()
        model.load_state_dict(
            torch.load(
                f'{cache_path}/{WEIGHT_FOLDER_NAME}/rd64-uni.pth',
                map_location=torch.device('cuda'),
            ),
            strict=False,
        )
        self.model = model

        self.transformation = self.default_transformation()

    @staticmethod
    def document_to_pil(doc: Document) -> Image:
        uri_data = urlopen(doc.uri)
        return Image.open(BytesIO(uri_data.read()))

    @staticmethod
    def default_transformation() -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512)),
        ])

    @requests(on='/segment')
    def segment(self, docs: DocumentArray, parameters: Dict, **kwargs):
        '''
        Parameters for CLIP segmentation:

        Document.text: Prompt for segmentation.
        @parameters.adaptive_thresh_block_size: Adaptive thresholding blocksize,
          as integer.
        @parameters.adaptive_thresh_c: Adaptive thresholding c value, as float.
        @parameters.binary_thresh_strength: Strength of binary thresholding,
          lower = more promiscuous.
        @parameters.thresholding_type: Type of thresholding, default binary
          method.
        '''
        request_time = time.time()

        # Parse parameters.
        invert = parameters.get('invert', False)
        try:
            thresholding_type = parameters.get('thresholding_type',
                THRESHOLDING_METHODS.BINARY.value)
            thresholding_type = THRESHOLDING_METHODS(thresholding_type)
        except ValueError:
            thresholding_type = THRESHOLDING_METHODS.BINARY

        adaptive_thresh_block_size = None
        adaptive_thresh_c = None
        binary_thresh_strength = None
        if thresholding_type == THRESHOLDING_METHODS.BINARY:
            binary_thresh_strength = parameters.get('binary_thresh_strength',
                THRESHOLD_BINARY_DEFAULT_STRENGTH_VALUE)
            try:
                binary_thresh_strength = int(binary_thresh_strength)
            except Exception:
                pass
            if not isinstance(binary_thresh_strength, int):
                binary_thresh_strength = THRESHOLD_BINARY_DEFAULT_STRENGTH_VALUE
        if thresholding_type == THRESHOLDING_METHODS.ADAPTIVE_MEAN or \
            thresholding_type == THRESHOLDING_METHODS.ADAPTIVE_GAUSSIAN:
            adaptive_thresh_block_size = parameters.get(
                'adaptive_thresh_block_size',
                THRESHOLD_ADAPTIVE_DEFAULT_BLOCK_SIZE)
            try:
                adaptive_thresh_block_size = int(adaptive_thresh_block_size)
            except Exception:
                pass
            if not isinstance(adaptive_thresh_block_size, int):
                adaptive_thresh_block_size = \
                    THRESHOLD_ADAPTIVE_DEFAULT_BLOCK_SIZE
            if adaptive_thresh_block_size % 2 != 1:
                adaptive_thresh_block_size -= 1
            adaptive_thresh_c = parameters.get(
                'adaptive_thresh_c',
                THRESHOLD_ADAPTIVE_DEFAULT_C)
            if not isinstance(adaptive_thresh_c, float):
                adaptive_thresh_c = THRESHOLD_ADAPTIVE_DEFAULT_C

        with torch.no_grad():
            for doc in docs:
                prompts = [doc.text]
                image_in = self.document_to_pil(doc)
                image_in = image_in.convert('RGB')
                image_unsqueezed = self.transformation(image_in).unsqueeze(0)

                mask_preds = self.model(image_unsqueezed.repeat(1,1,1,1),
                    prompts)[0]
                sigmoidy = torch.sigmoid(mask_preds[0][0]).cpu().detach().numpy()
                mask_as_arr = (sigmoidy * 255 / np.max(sigmoidy)).astype('uint8')
                image_mask_init = Image.fromarray(mask_as_arr)
                mask_cv = cv2.cvtColor(np.array(image_mask_init),
                    cv2.COLOR_RGB2BGR)
                gray_image = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)

                # Fallthrough (THRESHOLDING_METHODS.NONE) is just the gray
                # image.
                bw_image = gray_image
                if thresholding_type == THRESHOLDING_METHODS.BINARY:
                    (_, bw_image) = cv2.threshold(
                        gray_image,
                        binary_thresh_strength,
                        255,
                        cv2.THRESH_BINARY,
                    )
                if thresholding_type == THRESHOLDING_METHODS.ADAPTIVE_MEAN or \
                    thresholding_type == THRESHOLDING_METHODS.ADAPTIVE_GAUSSIAN:
                    a_method = cv2.ADAPTIVE_THRESH_MEAN_C
                    if thresholding_type == THRESHOLDING_METHODS.ADAPTIVE_GAUSSIAN:
                        a_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                    bw_image = cv2.adaptiveThreshold(
                        gray_image,
                        255,
                        a_method,
                        cv2.THRESH_BINARY,
                        adaptive_thresh_block_size,
                        adaptive_thresh_c,
                    )

                cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
                image_mask = Image.fromarray(bw_image) \
                    .convert('L') \
                    .resize(image_in.size, Image.NEAREST)
                # Normally the mask "selects" the query with the alpha layer,
                # but if invert is on it selects the opposite.
                if not invert:
                    image_mask = ImageOps.invert(image_mask)
                image_rgba = image_in.copy()
                image_rgba.putalpha(image_mask)

                buffered = BytesIO()
                image_rgba.save(buffered, format='PNG')
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                    tags={
                        'request': {
                            'api': 'segment',
                            'adaptive_thresh_block_size': adaptive_thresh_block_size,
                            'adaptive_thresh_c': adaptive_thresh_c,
                            'binary_thresh_strength': binary_thresh_strength,
                            'invert': invert,
                            'thresholding_type': thresholding_type.value,
                        },
                        'text': doc.text,
                        'generator': 'clipseg',
                        'request_time': request_time,
                        'created_time': time.time(),
                    },
                ).convert_blob_to_datauri()
                _d.text = doc.text
                doc.matches.append(_d)

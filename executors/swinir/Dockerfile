FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget git python3.8 python3-pip \
    && ln -sf python3.8 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workdir

RUN git clone --depth=1 https://github.com/jina-ai/SwinIR.git  && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    cd SwinIR && pip install --timeout=1000 -e . && cd -

COPY . /workdir/executor
RUN pip install --no-cache-dir -r ./executor/requirements.txt

WORKDIR /workdir/executor
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
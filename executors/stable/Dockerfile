FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget git python3.8 python3-pip \
    && ln -sf python3.8 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workdir

COPY . /workdir/executor
RUN pip install --no-cache-dir -r ./executor/requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR /workdir/executor
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
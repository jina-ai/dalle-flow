FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget git python3 python3-pip \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workdir

COPY . /workdir/executor
RUN pip install --no-cache-dir -r ./executor/requirements.txt
RUN pip install jax[cuda11_cudnn82]==0.3.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# ARG USER_ID=1000
# ARG GROUP_ID=1000

# ARG USER_NAME=dalle
# ARG GROUP_NAME=dalle

# RUN groupadd -g ${GROUP_ID} ${USER_NAME} &&\
#     useradd -l -u ${USER_ID} -g ${USER_NAME} ${GROUP_NAME} &&\
#     mkdir /home/${USER_NAME} &&\
#     chown ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME} &&\
#     chown -R ${USER_NAME}:${GROUP_NAME} /workdir/

# USER ${USER_NAME}

WORKDIR /workdir/executor
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]

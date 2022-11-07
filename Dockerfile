FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# given by builder
ARG PIP_TAG
# something like "gcc libc-dev make libatlas-base-dev ruby-dev"
ARG APT_PACKAGES="git wget"

WORKDIR /dalle

ADD requirements.txt dalle-flow/
ADD flow.yml dalle-flow/
ADD flow_parser.py dalle-flow/
ADD start.sh dalle-flow/

RUN chmod +x dalle-flow/start.sh

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install python3.10 python3.10-dev -y \
    && apt-get install -y --no-install-recommends sudo python3 python3-pip wget apt-utils libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install --upgrade virtualenv \
    && pip install wheel setuptools

RUN if [ -n "${APT_PACKAGES}" ]; then apt-get update && apt-get install --no-install-recommends -y ${APT_PACKAGES}; fi && \
    git clone --depth=1 https://github.com/jina-ai/SwinIR.git  && \
    git clone --depth=1 https://github.com/CompVis/latent-diffusion.git && \
    git clone --depth=1 https://github.com/jina-ai/glid-3-xl.git && \
    git clone --depth=1 --branch v0.0.15 https://github.com/AmericanPresidentJimmyCarter/stable-diffusion.git && \
    cd dalle-flow && python3 -m virtualenv --python=/usr/bin/python3.10 env && . env/bin/activate && cd - && \
    pip install --upgrade cython && \
    pip install --upgrade pyyaml && \
    pip install basicsr facexlib gfpgan && \
    pip install realesrgan && \
    git clone --depth=1 https://github.com/timojl/clipseg.git && \
    pip install jax[cuda11_cudnn82]==0.3.15 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install PyYAML numpy tqdm pytorch_lightning einops numpy omegaconf && \
    pip install https://github.com/crowsonkb/k-diffusion/archive/master.zip && \
    cd latent-diffusion && pip install --timeout=1000 -e . && cd - && \
    cd glid-3-xl && pip install --timeout=1000 -e . && cd - && \
    cd dalle-flow && pip install --timeout=1000 --compile -r requirements.txt && cd - && \
    cd stable-diffusion && pip install --timeout=1000 -e . && cd - && \
    cd SwinIR && pip install --timeout=1000 -e . && cd - && \
    cd clipseg && pip install --timeout=1000 -e . && cd - && \
    cd glid-3-xl && \
    # now remove apt packages
    if [ -n "${APT_PACKAGES}" ]; then apt-get remove -y --auto-remove ${APT_PACKAGES} && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*; fi

COPY executors dalle-flow/executors
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

ARG USER_ID=1000
ARG GROUP_ID=1000

ARG USER_NAME=dalle
ARG GROUP_NAME=dalle

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${USER_NAME} ${GROUP_NAME} | chpasswd && \
    adduser ${USER_NAME} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    mkdir /home/${USER_NAME} && \
    chown ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME} && \
    chown -R ${USER_NAME}:${GROUP_NAME} /dalle/

USER ${USER_NAME}

WORKDIR /dalle/dalle-flow

ENTRYPOINT ["./start.sh"]

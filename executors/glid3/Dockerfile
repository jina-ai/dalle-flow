FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget git python3.8 python3-pip \
    && ln -sf python3.8 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workdir

RUN git clone --depth=1 https://github.com/CompVis/latent-diffusion.git && \
    git clone --depth=1 https://github.com/hanxiao/glid-3-xl.git && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    cd latent-diffusion && pip install --timeout=1000 -e . && cd - && \
    cd glid-3-xl && pip install --timeout=1000 -e . && cd - && \
    cd glid-3-xl && \
    wget -q https://dall-3.com/models/glid-3-xl/bert.pt &&  \
    wget -q https://dall-3.com/models/glid-3-xl/kl-f8.pt &&  \
    wget -q https://dall-3.com/models/glid-3-xl/finetune.pt && cd -

COPY . /workdir/executor
RUN pip install --no-cache-dir -r ./executor/requirements.txt

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

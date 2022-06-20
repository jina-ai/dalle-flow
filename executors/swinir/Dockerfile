FROM jinaai/jina:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workdir

RUN git clone --depth=1 https://github.com/JingyunLiang/SwinIR.git

COPY . /workdir/executor
RUN pip install --no-cache-dir -r ./executor/requirements.txt

ARG USER_ID=1000
ARG GROUP_ID=1000

ARG USER_NAME=dalle
ARG GROUP_NAME=dalle

RUN groupadd -g ${GROUP_ID} ${USER_NAME} &&\
    useradd -l -u ${USER_ID} -g ${USER_NAME} ${GROUP_NAME} &&\
    mkdir /home/${USER_NAME} &&\
    chown ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME} &&\
    chown -R ${USER_NAME}:${GROUP_NAME} /workdir/

USER ${USER_NAME}

WORKDIR /workdir/executor
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]

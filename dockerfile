FROM ubuntu:focal

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    python-is-python3 \
    python3-dev \
    build-essential \
    pip \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 \
    -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

WORKDIR /home/ofa_telegram 

RUN git clone https://github.com/pytorch/fairseq.git && \
    git clone https://github.com/OFA-Sys/OFA.git 

COPY requirements.txt ./

RUN pip install -r requirements.txt && \
    pip install -r OFA/requirements.txt && \
    cd fairseq && python -m pip install fairseq --use-feature=in-tree-build ./ && \ 
    python -m pip cache purge

ADD https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt \
    ./OFA/checkpoints/caption.pt

WORKDIR OFA
COPY ofa.py ./

ARG TG_TOKEN=""
ARG MY_CHAT=""

ENV TG_TOKEN="${TG_TOKEN}" 
ENV MY_CHAT="${MY_CHAT}" 

# Configure container startup
ENTRYPOINT ["python", "ofa.py"]
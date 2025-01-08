#FROM python:3.12-slim
# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && python --version \
    && pip --version \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY deploy ./deploy
COPY main.py .
COPY train.py .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
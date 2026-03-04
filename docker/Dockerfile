ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION
ARG USER_ID

# metainformation
LABEL org.opencontainers.image.version = "1.0.0"
LABEL org.opencontainers.image.source = "https://github.com/atonderski/gasp"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

## CUDA architectures required by custom CUDA kernels.
## To speed up the build, remove all architectures except the one for your GPU.
## See: https://developer.nvidia.com/cuda-gpus (e.g. 6.1 for TITAN Xp / Pascal)
ARG CUDA_ARCHITECTURES=90;89;86;80;75;61

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    vim-tiny \
    wget \
    less \
    && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip "setuptools<70.0" pathtools promise pybind11
SHELL ["/bin/bash", "-c"]

# Install PyTorch.
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3.10 -m pip install --no-cache-dir \
    torch==2.4.1+cu${CUDA_VER} \
    torchvision==0.19.1+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

# Setup SSH known hosts for GitHub.
RUN mkdir ~/.ssh && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# Add source code to container.
ADD . /gasp

# Install pose interpolation CUDA extension (used by occupancy sampler).
# TORCH_CUDA_ARCH_LIST must be space-separated (e.g. "9.0 8.9 8.6 8.0 7.5") for PyTorch.
WORKDIR /gasp/gasp/model/ops
RUN TORCH_CUDA_ARCH_LIST=$(echo "${CUDA_ARCHITECTURES}" | sed -E 's/([0-9])([0-9])/\1.\2/g' | tr ';' ' ') \
    python3.10 setup.py build install

# Install transformers for DINOv2 feature extraction.
RUN python3.10 -m pip install git+https://github.com/huggingface/transformers.git
RUN python3.10 -m pip install accelerate

# Install Python requirements and the gasp package.
WORKDIR /gasp
RUN python3.10 -m pip install --no-cache-dir -r requirements/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -e .

CMD /bin/bash -l

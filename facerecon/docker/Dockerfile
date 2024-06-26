# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Note: Should also work with NVIDIA's Docker image builds such as
#
# nvcr.io/nvidia/pytorch:20.09-py3
#
# This file defaults to pytorch/pytorch as it works on slightly older
# driver versions.
FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

# Modifications to Dockerfile begin here
# Source Dockerfile: https://github.com/NVlabs/nvdiffrast/blob/main/docker/Dockerfile
# Retrieved 4/26/2024
COPY facerecon/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Install nvdiffrast
RUN pip install --upgrade pip
RUN pip install ninja imageio imageio-ffmpeg
RUN git clone https://github.com/NVlabs/nvdiffrast.git
RUN mkdir -p /tmp/pip
RUN cp -r nvdiffrast/nvdiffrast/ /tmp/pip/
RUN cp nvdiffrast/setup.py /tmp/pip/
RUN cp nvdiffrast/README.md /tmp/pip/
RUN cd /tmp/pip && pip install .
WORKDIR /workspace/
RUN pip install kornia dominate trimesh tensorflow scikit-image
# Clone Deep3DFaceRecon_pytorch repo and cd into it
RUN git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
RUN cd Deep3DFaceRecon_pytorch/  # /workspace/Deep3DFaceRecon_pytorch
# Clone insightface repo and copy model over to Deep3DFaceRecon_pytorch models dir
RUN git clone https://github.com/deepinsight/insightface.git
RUN cp -r /workspace/insightface/recognition/arcface_torch /workspace/Deep3DFaceRecon_pytorch/models/
# Copy over models
COPY data/pretrained/facerecon/BFM/01_MorphableModel.mat /workspace/Deep3DFaceRecon_pytorch/BFM/
COPY data/pretrained/facerecon/BFM/Exp_Pca.bin /workspace/Deep3DFaceRecon_pytorch/BFM/
COPY data/pretrained/facerecon/checkpoints/facerecon_20230425/ /workspace/Deep3DFaceRecon_pytorch/checkpoints/facerecon_20230425/
# Copy over runner file
COPY facerecon/facerecon_deep3dfacerecon.py /workspace/Deep3DFaceRecon_pytorch/
# Install modules specific to modification of Deep3dFaceRecon_pytorch
RUN pip install loguru
WORKDIR /workspace/Deep3DFaceRecon_pytorch/

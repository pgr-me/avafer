FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update
RUN python -m pip install --upgrade pip
WORKDIR /workspace
RUN git clone https://github.com/TropComplique/mtcnn-pytorch.git
WORKDIR mtcnn-pytorch


FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update && apt-get install -y vim
RUN python -m pip install --upgrade pip
RUN pip install loguru tqdm ipdb
WORKDIR /workspace
RUN git clone https://github.com/TropComplique/mtcnn-pytorch.git
WORKDIR /workspace/mtcnn-pytorch
COPY facelandmarks/facelandmarks_mtcnn.py /workspace/mtcnn-pytorch/


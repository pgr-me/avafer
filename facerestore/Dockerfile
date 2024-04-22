FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git git-lfs vim

RUN python -m pip install --upgrade pip
RUN pip install diffusers["torch"] transformers scikit-image
RUN git lfs install
RUN git clone https://huggingface.co/spaces/nightfury/Colorizer_Models 
RUN conda install -y conda-forge::conda-build
RUN conda-develop /workspace/Colorizer_Models/
COPY facerestore.py /workspace/

# Facial Expression Reconstruction

This readme documents how to perform 3D facial reconstruction as one component of the work of this project.

The code in this directory is based on the [nvdiffrast repo](https://github.com/NVlabs/nvdiffrast).

## Setup

* We performed the work of this project using the following AWS AMI: Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 20.04) 20240410 and used a g5.xlarge instance type.
* Create the image:
```shell
$ docker build -t avafer:latest -f docker/Dockerfile .
```
* Create the container:
```shell
docker run -dit --gpus all --name avafer avafer:latest
```
* Prepare models used for inference by following the instructions from the [Deep3DFaceRecon_pytorch readme](https://github.com/sicxu/Deep3DFaceRecon_pytorch?tab=readme-ov-file#inference-with-a-pre-trained-model). 



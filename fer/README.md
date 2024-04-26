
# Facial Expression Recognition

This readme documents how to classify facial expressions (i.e., perform facial expression recognition).

This module is limited to the [RMN](https://github.com/phamquiluan/ResidualMaskingNetwork) pretrained network but we plan to add more pretrained networks.

## Setup

* We performed the work of this project using the following AWS AMI: Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 20.04) 20240410 and used a g5.xlarge instance type.
* If you just want to run this container, do the following:
* Change directory into `fer/`
```shell
$ cd fer/
```
### Running RMN

* Create the image:
```shell
$ docker build -t fer-rmn:latest -f RMN-Dockerfile .
```
* Create the container:
```shell
docker run -dit --gpus all --name fer-rmn fer-rmn:latest
```
* Enter the container:
```shell
$ docker exec -ti fer-rmn bash
```
* In the container, execute the following command to reproduce help menu:
```
$ python fer_rmn.py -h
```


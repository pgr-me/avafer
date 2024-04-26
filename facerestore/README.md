# Facial restoration

## Setup

Make sure you're in the `facerestore` directory when you run the `docker build` and `docker run` commands.

## Build the image.
```shell
$ docker build -t facerestore:latest -f facerestore/Dockerfile .
```

## Create the container
```shell
$ docker run -v /absolute/path/to/data/:/data -dit --name facerestore --gpus all facerestore:latest
```

## Enter the container.
```shell
$ docker exec -ti facerestore bash
```

## Execute the face restoration script and display help menu.
```shell
$ python facerestore.py -h
```

## Example command to process test (i.e., example) benchmark:
```shell
$ python facerestore.py --root_dir /data --benchmark test_dataset
```

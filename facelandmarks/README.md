# Face Landmarks

## Setup

Make sure you're in the root directory of this repo when running the `docker build` and `docker run` commands.

## Build the image.
```shell
$ docker build -t facelandmarks:latest -f facelandmarks/Dockerfile .
```

## Create the container
```shell
$ docker run -v /absolute/path/to/data/:/data -dit --name facelandmarks --gpus all facelandmarks:latest
```

## Enter the container.
```shell
$ docker exec -ti facelandmarks bash
```

## Execute the face restoration script and display help menu.
```shell
$ python get_facelandmarks.py -h
```

## Example command to process test (i.e., example) benchmark:
```shell
$ python get_facelandmarks.py --root_dir /data --benchmark test_dataset
```

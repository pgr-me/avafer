# Facial restoration

## Build the image.
```shell
$ docker build -t facerestore:latest -f Dockerfile .
```

## Create the container.
```shell
$ docker run -v /local/path/to/imgs:/workspace/data/raw -dit --name facerestore --gpus all facerestore:latest
```

## Enter the container.
```shell
$ docker exec -ti facerestore bash
```


## Execute the face restoration script and display help menu.
```shell
$ python facerestore.py -h
```

#!/bin/sh
docker run -it --gpus all --shm-size 64G \
    -v /path/to/this/repository:/workspace/ \
    pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime bash


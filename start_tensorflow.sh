#!/bin/bash
docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/:/home/ tensorflow/tensorflow:latest-py3-lab

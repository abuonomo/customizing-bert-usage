#!/bin/bash
#docker run -it -p 8887:8888 -p 6007:6006 \
#    -v $(pwd)/:/home/ abuonomo/tensorflow
docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/:/home/ tensorflow/tensorflow:latest-gpu-py3-lab

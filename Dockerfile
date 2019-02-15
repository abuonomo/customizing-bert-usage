FROM tensorflow/tensorflow:latest-py3

RUN pip install jupyterlab && \
    pip install pyyaml && \
    pip install tqdm

EXPOSE 8888
EXPOSE 6006
WORKDIR /home/
ENV PYTHONPATH ="PYTHONPATH:/home/bert/:/home/src/"
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]


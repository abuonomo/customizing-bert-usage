FROM tensorflow/tensorflow:latest-py3

RUN pip install jupyterlab, pyyaml, tqdm

EXPOSE 8888
WORKDIR /home/
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"] 


# How to Use BERT

1) Clone this repo.
2) Download the BERT model for your task. You can see information on a variety of BERT models [here](https://github.com/google-research/bert). I have tested this project using [BERT-Base, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip). Place these models in the root directory of this repo (be sure not to commit the models later!). 
3) Build the docker image with necessary dependenices by executing:  
    `docker build -t tensorflow/tensorflow:latest-py3-lab .`  
    You can then run:  
    `docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/:/home/ tensorflow/tensorflow:latest-py3-lab`  
    to start the container. Then, either bash into the container, or start a terminal using the jupyterlab interface.
4) Generate train, dev, and test .tsv files. You can run [this_script](test/example_scripts/make_formatted_sets.sh) to generate correctly formatted example tsv datasets. This script uses the example input dataset [here](test/datasets/results_small.json).  
** keep in mind that example tsv files generated with this script will likely have too few data points in order to be used by BERT without encountering an error.
5) Once you have an adequately large dataset, see [this script](test/example_scripts/train_and_eval.sh). It looks like this:
    ```bash
    python ../../src/helper_classes.py ../data/engine_tests/ \
        ../../cased_L-12_H-768_A-12/bert_config.json \
        ../../cased_L-12_H-768_A-12/vocab.txt \
        ../data/engine_tests/tf --do_train --do_eval
    ```
    You can use `python helper_classes.py -h` to see information about the script parameters. Essentially, you will always have to tailor the first argument `../data/engine_tests/`, the fourth argument `../data/engine_tests/tf`, and the flags for your purposes.  The first argument is the directory where your `train.tsv`, `dev.tsv`, and `test.tsv` files reside. The fourth argument is the directory where your fine-tuned BERT models will be placed during training. This empty folder must be created ahead of time.
6) Once training is complete, you can now use your models for new predictions. See [this script](test/example_scripts/predict.sh) for an example. It looks like this:
    ```bash
    #!/usr/bin/env bash
    python ../../src/helper_classes.py ../data/engine_tests/ \
        ../../cased_L-12_H-768_A-12/bert_config.json \
        ../../cased_L-12_H-768_A-12/vocab.txt \
        ../data/engine_tests/tf --do_predict
    ```   
    Again, the first argument, fourth argument, and flags must be tailored for your use case. The first argument remains the directory in which your train.tsv, test.tsv, and dev.tsv files reside. For this prediction script, it is only necessary that `test.tsv` resides in this folder. The fourth argument `../data/engine_tests/tf` is the directory where your fine tuned BERT models where placed step 4. This script will produce an output at `../data/engine_tests/tf/test_results.tsv` which will contain the results of this prediction.

## Using a GPU
In order to fine-tune using BERT with a significantly-sized dataset, you may want to use a GPU. If you do this, you will need to build a different docker image. You can do so by running:  
 `docker build -f Dockerfile-gpu -t tensorflow/tensorflow:latest-gpu-py3-lab`.   
 Then start the container using:  
 `nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/:/home/ tensorflow/tensorflow:latest-gpu-py3-lab`.   
 This implementation uses nvidia-docker and is only tested with nvidia-docker.  

## Jupyter notebook for interactive predictions

After you create your models, you may want to obtain predictions interactively in a jupyter notebook. You can see an example of this [here](notebooks/do_predictions.ipynb).

use script to make model  
    - clarify which folder for what, where do you manipulate params  
use model to make predictions

use model to make multiple predictions

add bert pythonpath
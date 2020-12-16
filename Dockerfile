#
# neuron-sdk
# @author Loreto Parisi (loreto at musixmatch dot com)
# Copyright (c) 2020 Loreto Parisi
#

FROM ubuntu:18.04

LABEL maintainer Loreto Parisi loreto@musixmatch.com

WORKDIR app



RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    wget \
    gpg

RUN apt -y install python3.7

# jupyter
#RUN pip install jupyterlab && \
#    pip install notebook


# pip config
RUN mkdir ~/.pip/ && \
    touch ~/.pip/pip.conf && \
    echo [global] >> ~/.pip/pip.conf && \
    echo extra-index-url = https://pip.repos.neuron.amazonaws.com >> ~/.pip/pip.conf && \
    cat ~/.pip/pip.conf


# wheels
RUN pip3 download --no-deps neuron-cc && \
    wget https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-1.0.24045.0%2B13ab1a114-cp37-cp37m-linux_x86_64.whl


# tensorflow
RUN pip3 install neuron-cc && \
    pip3 install tensorflow-neuron && \
    pip3 install tensorboard-neuron && \
    tensorboard_neuron -h | grep run_neuron_profile


# pytorch
RUN pip3 install neuron-cc[tensorflow] && \
    pip3 install torch && \
    pip3 install torchvision==0.4.0 && \
    pip3 install torch-neuron

# Kubeflow config

ENV NB_PREFIX /

CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
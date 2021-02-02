# neuron-sdk-docker
AWS Neuron-SDK utilities for `Tensorflow` and `Pytorch` models compilation using `Docker` and Python `virtualenv`.

## Docker images
We user `Docker` as main build system. As alternative please use `virtualenv` in the next chapter.

### Build [docker]
We build the Docker container
```
docker build -f Dockerfile -t neuronsdk .
```

### Run [docker]
We run the Docker image with a attached `/src` volume for editable sources and a `/root` volumes for models files.
```
docker run --rm -it -d -v `pwd`/src:/app -v $VOLUME:/root --name neuronsdk-compile neuronsdk
docker exec -it neuronsdk-compile bash
```

### Docker images [docker]
The build Docker image is

```
REPOSITORY                         TAG                 IMAGE ID            CREATED             SIZE
neuronsdk                          latest              7cb529332e30        21 hours ago        7.63GB
```

## Virtualenv
To build and run in `virtualenv` please use the following

### Build [virtualenv]
Be sure to enter the docker container built before.
```
python3 -m venv test_venv
source test_venv/bin/activate
pip install -U pip
```

We need to setup `pip` in `virtualenv`

```
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```

Base SDK install
```
pip download --no-deps neuron-cc
# The above shows you the name of the package downloaded
# Use it in the following command
wget https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-<VERSION FROM FILE>.whl.asc
```

Example:

### Wheels [virtualenv]
To install the wheel choose the version as stated above:
```
https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-1.0.24045.0%2B13ab1a114-cp37-cp37m-linux_x86_64.whl
```

then

```
wget https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-1.0.24045.0%2B13ab1a114-cp37-cp37m-linux_x86_64.whl
```

To verify the package signature, run


### GPG Signature [virtualenv]

```
gpg --verify neuron_cc-1.0.24045.0%2B13ab1a114-cp37-cp37m-linux_x86_64.whl.asc neuron_cc-1.0.24045.0%2B13ab1a114-cp37-cp37m-linux_x86_64.whl
gpg: Signature made Wed Nov 18 02:00:56 2020 UTC
gpg:                using RSA key 5749CAD8646D9185
gpg: Good signature from "Amazon AWS Neuron <neuron-maintainers@amazon.com>" [unknown]
gpg: WARNING: This key is not certified with a trusted signature!
gpg:          There is no indication that the signature belongs to the owner.
Primary key fingerprint: 00FA 2C10 7926 0870 A76D  2C28 5749 CAD8 646D 9185
```

### Tensorflow [virtualenv]
```
pip install neuron-cc
pip install tensorflow-neuron
pip install tensorboard-neuron
tensorboard_neuron -h | grep run_neuron_profile
```

### Pytorch [virtualenv]
```
pip install neuron-cc[tensorflow]
pip install torch-neuron
```

## Dependencies
Tensorflow dependencies are
```
decorator, numpy, inferentia-hwm, dmlc-topi, attrs, dmlc-tvm, dmlc-nnvm, scipy, networkx, pycparser, cffi, six, islpy, neuron-cc
Successfully installed attrs-20.3.0 cffi-1.14.3 decorator-4.4.2 dmlc-nnvm-1.0.3544.0+0 dmlc-topi-1.0.3544.0+0 dmlc-tvm-1.0.3544.0+0 inferentia-hwm-1.0.1938.0+0 islpy-2018.2+aws2018.x.585.0.bld0 networkx-2.4 neuron-cc-1.0.24045.0+13ab1a114 numpy-1.18.4 pycparser-2.20 scipy-1.4.1 six-1.15.0
```

Pytorch dependencies are

```

```
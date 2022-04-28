# Supervised Classification with Born's Rule

## Installation

Install with:

```bash
pip install .
```

## Usage

### Scikit-Learn

```py
from bornrule import BornClassifier
```

- Use it like any other `sklearn` classifier
- Supports `scipy` sparse matrices 
- Supports `cupy` arrays and sparse matrices for GPU-accelerated computing

### PyTorch

```py
from bornrule.torch import Born
```
- Use it like any other `torch` layer
- Supports real and complex-valued inputs
- Output: probabilities in the range [0,1]

### SQL

```py
from bornrule.sql import BornClassifierSQL
```

- Use for in-database classification
- Supports inputs represented as json `{key: value, ...}`

## Paper replication

All the results in the paper are obtained using Python 3.9 on a Google Cloud Virtual Machine equipped with 
CentOS 7, 12 vCPU Intel Cascade Lake 85 GB RAM, 1 GPU NVIDIA Tesla A100, and CUDA 11.5.

### Poetry

If you are running a linux distribution with CUDA 11.5, use the following commands to:

- download [`poetry`](https://python-poetry.org)
- create a virtual environment with python 3.9
- install the dependencies
- run the replication code in `nips.py`

```commandline
pip install poetry
poetry env use python3.9
poetry install
poetry run python -u nips.py > nips.log &
```

### Pip

If you are not running a linux distribution with CUDA 11.5, use `pip` as follows.

Install this project:

```commandline
pip install .
```

Install additional dependencies to replicate the paper:

```commandline
pip install bs4==0.0.1 nltk==3.7 matplotlib==3.5.1
```

Install [`pytorch`](https://pytorch.org) version `1.11.0` with GPU support. For CUDA 11.5 the command is:
```commandline
pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
```

Install [`cupy`](https://docs.cupy.dev/en/stable/install.html) version `10.4.0`. For CUDA 11.5 the command is:
```commandline
pip install cupy-cuda115==10.4.0
```

Run the script `nips.py`:

```commandline
python -u nips.py > nips.log &
```

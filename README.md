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


Install this project with [`poetry`](https://python-poetry.org): 

```commandline
pip install poetry
poetry install
```

Install [`cupy`](https://docs.cupy.dev/en/stable/install.html). For CUDA 11.5 the command is:
```commandline
pip install cupy-cuda115
```

Run the script `nips.py`:

```commandline
python -u nips.py > nips.log &
```

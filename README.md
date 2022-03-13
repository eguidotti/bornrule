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

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the script `nips.py`. It takes several days to complete. 
You may consider modifying the parameters for a first run. 

All the results are obtained using Python 3.9 on a Google Cloud Virtual Machine equipped with 
CentOS 7, 12 vCPU Intel Cascade Lake 85 GB RAM, 1 GPU NVIDIA Tesla A100.

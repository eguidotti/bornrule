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



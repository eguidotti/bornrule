# Classification with Born's Rule

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" align="right" height="128"/>This repository contains the code for the paper [Text Classification with Born's Rule](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c88d0c9bea6230b518ce71268c8e49e0-Abstract-Conference.html). The classifier is implemented in python and available on [PyPI](https://pypi.org/project/bornrule/). The documentation is available [here](https://bornrule.eguidotti.com).

## Installation

Install via `pip` with:

```bash
pip install bornrule
```

## Usage

The package implements three versions of the classifier. The classification algorithm is compatible with the [scikit-learn](https://scikit-learn.org/) ecosystem. The neural version is compatible with [pytorch](https://pytorch.org/). The SQL version supports in-database classification.

### Scikit-Learn

```py
from bornrule import BornClassifier
```

- Use it as any other `sklearn` classifier
- Supports both dense and sparse input and GPU-accelerated computing via `cupy`

### PyTorch

```py
from bornrule.torch import Born
```

- Use it as any other `torch` layer
- Supports real and complex-valued inputs. Outputs probabilities in the range [0, 1]

### SQL

```py
from bornrule.sql import BornClassifierSQL
```

- Equivalent to the class  `BornClassifier`  but for in-database classification
- Supports inputs represented as json `{feature: value, ...}`

## Paper replication

All the results in the paper are obtained using Python 3.9 on a Google Cloud Virtual Machine equipped with 
CentOS 7, 12 vCPU Intel Cascade Lake 85 GB RAM, 1 GPU NVIDIA Tesla A100, and CUDA 11.5.

Install this package:

```commandline
pip install bornrule==0.1.0
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

The script generates a folder named `results` with all the results in the paper. Additional information are saved to the log file `nips.log`

## Cite as

>Emanuele Guidotti and Alfio Ferrara. Text Classification with Born's Rule. In *Advances in Neural Information Processing Systems*, volume 35, pages 30990–31001, 2022.

A BibTeX entry for LaTeX users is:

```bibtex
@inproceedings{guidotti2022text,
 title = {Text Classification with Born's Rule},
 author = {Guidotti, Emanuele and Ferrara, Alfio},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {30990--31001}, 
 volume = {35},
 year = {2022}
}
```


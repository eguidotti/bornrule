# Classification with Born's Rule

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" align="right" height="128"/>This repository implements the classifier proposed in:

> Emanuele Guidotti and Alfio Ferrara. Text Classification with Born’s Rule. *Advances in Neural Information Processing Systems*, 2022.

<div align="center">
  [<a href="https://openreview.net/pdf?id=sNcn-E3uPHA">Paper</a>] - 
  [<a href="https://nips.cc/media/neurips-2022/Slides/54723.pdf">Slides</a>] - 
  [<a href="https://nips.cc/media/PosterPDFs/NeurIPS%202022/8d7628dd7a710c8638dbd22d4421ee46.png">Poster</a>]
</div>

## Installation

```bash
pip install bornrule
```

## Usage

### Scikit-Learn

```py
from bornrule import BornClassifier
```

- Use it as any other `sklearn` classifier
- Supports `scipy` sparse matrices, and `cupy` arrays for GPU-accelerated computing
- Documentation available [here](https://eguidotti.github.io/bornrule/sklearn/)

### PyTorch

```py
from bornrule.torch import Born
```
- Use it as any other `torch` layer
- Supports real and complex-valued inputs. Outputs probabilities in the range [0, 1]
- Documentation available [here](https://eguidotti.github.io/bornrule/pytorch/)

### SQL

```py
from bornrule.sql import BornClassifierSQL
```

- Use it for in-database classification
- Supports inputs represented as json `{feature: value, ...}`
- Documentation available [here](https://eguidotti.github.io/bornrule/sql/)

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

Please cite the following when using this software:

```bibtex
@inproceedings{guidotti2022text,
  title={Text Classification with Born's Rule},
  author={Emanuele Guidotti and Alfio Ferrara},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=sNcn-E3uPHA}
}
```


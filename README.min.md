<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" align="right" height="128"/>This package implements the classifier proposed in:

> Emanuele Guidotti and Alfio Ferrara. Text Classification with Born’s Rule. *Advances in Neural Information Processing Systems*, 2022.

<div align="center">
  [<a href="https://openreview.net/pdf?id=sNcn-E3uPHA">Paper</a>] -
  [<a href="https://nips.cc/media/PosterPDFs/NeurIPS%202022/8d7628dd7a710c8638dbd22d4421ee46.png">Poster</a>] - 
  [<a href="https://bornrule.eguidotti.com">Docs</a>]
</div>

## Usage

### Scikit-Learn

```py
from bornrule import BornClassifier
```

- Use it as any other `sklearn` classifier
- Supports both dense and sparse input and GPU-accelerated computing via `cupy`
- Documentation available [here](https://bornrule.eguidotti.com/sklearn/)

### PyTorch

```py
from bornrule.torch import Born
```

- Use it as any other `torch` layer
- Supports real and complex-valued inputs. Outputs probabilities in the range [0, 1]
- Documentation available [here](https://bornrule.eguidotti.com/pytorch/)

### SQL

```py
from bornrule.sql import BornClassifierSQL
```

- Use it for in-database classification
- Supports inputs represented as json `{feature: value, ...}`
- Documentation available [here](https://bornrule.eguidotti.com/sql/)

## Paper replication

The replication code is available at https://github.com/eguidotti/bornrule

## Cite as

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

# Classification with Born's Rule

This repository contains the code for the papers [Text Classification with Born's Rule](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c88d0c9bea6230b518ce71268c8e49e0-Abstract-Conference.html) and In-Database Text Classification with BornSQL (EDBT 2026, forthcoming). The code to reproduce the papers is available in the folders `nips` and `edbt`, respectively. The classifier is implemented in python and available on [PyPI](https://pypi.org/project/bornrule/). The complete documentation is available [here](https://bornrule.eguidotti.com).

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

## Cite as

>Emanuele Guidotti and Alfio Ferrara. Text Classification with Born's Rule. In *Advances in Neural Information Processing Systems*, volume 35, pages 30990–31001, 2022.

A BibTeX entry for LaTeX users is:

```bibtex
@inproceedings{guidotti2022born,
 title     = {Text Classification with Born's Rule},
 author    = {Guidotti, Emanuele and Ferrara, Alfio},
 booktitle = {Advances in Neural Information Processing Systems},
 pages     = {30990--31001}, 
 volume    = {35},
 year      = {2022}
}
```

>Emanuele Guidotti, Darya Shlyk, Stefano Montanelli, and Alfio Ferrara. In-Database Text Classification with BornSQL. *Proceedings of the 29th international conference on extending database technology*, forthcoming, 2026.

A BibTeX entry for LaTeX users is:

```bibtex
@inproceedings{guidotti2026bornsql,
  author    = {Emanuele Guidotti and Darya Shlyk and Stefano Montanelli and Alfio Ferrara},
  title     = {In-Database Text Classification with BornSQL},
  booktitle = {Proceedings of the 29th International Conference on Extending Database Technology (EDBT)},
  year      = {2026},
  note      = {Forthcoming}
}
```


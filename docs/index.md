# Get Started

<style>
  @media (max-width: 767px) {
    .hidden-mobile {
      display: none;
    }
  }
</style>

<img 
    class="hidden-mobile" 
    style="float:right; height:120px" 
    src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" 
/>

This website contains the documentation for the package `bornrule` available on [PyPI](https://pypi.org/project/bornrule/). The package implements the classifier proposed in the paper [Text Classification with Born's Rule](https://proceedings.neurips.cc/paper_files/paper/2022/file/c88d0c9bea6230b518ce71268c8e49e0-Paper-Conference.pdf). All code is available at the [GitHub repository](https://github.com/eguidotti/bornrule). 

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
@inproceedings{guidotti2022text,
 title = {Text Classification with Born's Rule},
 author = {Guidotti, Emanuele and Ferrara, Alfio},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {30990--31001}, 
 volume = {35},
 year = {2022}
}
```


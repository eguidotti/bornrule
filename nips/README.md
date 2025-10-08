
# Paper replication

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

Run the script `main.py`:

```commandline
python main.py
```

The script generates a folder named `results` with all the results in the paper.

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

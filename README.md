# MaLC: Manifold for Linguistic Complexity
## Overview


## Repository Structure
```
../
├README.md
├code
│   ├bottleneck.py
│   ├classes.py
│   ├curvature.py
│   ├pdgm.py
│   ├setup.py
│   └wassersetin.py
├data
│   ├bert
│   │   ├embedding-lang1.npy
│   │   ├embedding-lang2.npy
│   │   └...
│   └fasttext
│   │   ├embedding-lang1.npy
│   │   ├embedding-lang2.npy
│   │   └...
└output
    ├bert
    │   ├pdgm
    │   │   ├h0
    │   │   ├h1
    │   │   └...
    │   ├wasserstein.csv
    │   └bottleneck.csv
    └fasttext
        ├pdgm
        │   ├h0
        │   ├h1
        │   └...
        ├wasserstein.csv
        └bottleneck.csv
```

- `setup.py`: for preprocessing of the repository.

## How to Run
### Setup
Setting up will be processed by the command below in this repository:
```
pip install .
```

### Command Line & Options
Embedding can be given by the command below:
```
python embed.py 'lang' 'token'
```


|Argument|Function|
|-----|-----|
|--gpu|`bool` uses GPU|
|--k|`int` number of neighbor points|
|--d|`int` dimension the dataset is compressed into|
|--save_emb|`bool` saves the embeddings|

### Sample Dataset
#### fasttext


#### bert

## Citation
```
@dissertation{nakayama-2026-dissertation,
    author      =   "Nakayama, Takuto"
    year        =   "2026",
    title       =   "",
    university  =   "Keio University"
}
```
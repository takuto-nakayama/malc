# MaLC: Manifold for Linguistic Complexity
## Overview


## Repository Structure
```
malc/
├README.md
├python
│   ├classes.py
│   ├curvature.py
│   ├embed.py
│   └setup.py
│
├sample-data
│   ├embedding-en.h5
│   ├embedding-de.h5
│   └embedding-ja.h5
│
└output
    ├hoo
    ├foo
    └
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


## Citation
```
@dissertation{nakayama-2026-dissertation,
    author      =   "Nakayama, Takuto"
    year        =   "2026",
    title       =   "",
    university  =   "Keio University"
}
```
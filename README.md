# MaLC: Manifold for Linguistic Complexity
## Overview



## How to Run
### Setup
Setting up will be processed by the command below in this repository:
```
pip install .
```

### Word Manifold
#### to run
```
python word-manifold.py 'lang' 'text_path'
```
- `lang`: available language is shown in the table below
- `text_path`: the file path to the input text (whose lines are recommended to be corresponding to a certain text)

#### Options
|option|function|
|-----|-----|
|--save_path|`bool`; the result will be output in .csv;<br> default="wm-{lang}-{random_number}.csv"|
|--window_size|`int`; window size of n-gram; default=5|


## Sample Data
### fasttext


### bert


### Universal Declaration of Human Rights


### Gospels


## Citation
```
@dissertation{nakayama-2026-dissertation,
    author      =   "Nakayama, Takuto"
    year        =   "2026",
    title       =   "",
    university  =   "Keio University"
}
```
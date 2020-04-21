[![PyPI](https://img.shields.io/pypi/v/dl-cliche)](https://pypi.org/project/dl-cliche/)
![PyPI - License](https://img.shields.io/pypi/l/dl-cliche)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dl-cliche)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/dl-cliche.svg)](https://pypistats.org/packages/dl-cliche)

# dl-cliche: Packaging cliche utilities

This is a python module started to package utility functions that are locally made but used everyday.
After while being improved along with other projects, these short-cut functions have become essential to work with various project.

## Installation

```shell
pip install dl-cliche
```

## What's for?

You can avoid repeating yourself with these cliches:

```python
%matplotlib inline
%reload_ext autoreload
%autoreload 2
```

with following one liner.

```python
from dlcliche.notebook import *
```

## Quickstart for a python code

```python
from dlcliche.utils import *
```

Then you can simply start your code that uses numpy, pandas or utility like:

- `get_logger()` to make default logger instance.
- `lock_file_mutex()` to make OS dependency-free mutex lock.
- and so on...

## Quickstart for a jupyter notebook

```python
from dlcliche.notebook import *
from dlcliche.utils import *
```

## Warnings to ignore

Type followings where you want to suppress warnings later on.

```python
import warnings; simply_ignore(warnings)
```

Still need to import warnings, `simply_ignore` wraps what to do then.

## Documents

- [ ] Basic function interface document to be ready.
- [x] (Japanese) [機械学習個人レベルのワークフロー改善@Qiita](https://qiita.com/daisukelab/items/109812791d369891b812)

# dl-cliche: Packaging cliche utilities

This is a python module started to package utility functions locally made but used everyday.
And while being improved along with other projects, these short-cut functions have become essential to work with various project.

Now this is pip ready.

## Installation

```
pip install dlcliche
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

## Documents

- [ ] Basic function interface document to be ready.
- [x] (Japanese) [機械学習個人レベルのワークフロー改善@Qiita](https://qiita.com/daisukelab/items/109812791d369891b812)

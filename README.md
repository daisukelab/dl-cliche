# dl-cliche: A small python module to summarize all cliche codes

For Japanese: 日本語で解説記事を用意しました。 ⇒ [機械学習個人レベルのワークフロー改善@Qiita](https://qiita.com/daisukelab/items/109812791d369891b812)

This is a python module created for local use, but also addresses widely common tiny issues.

You can replace cliche like this:

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


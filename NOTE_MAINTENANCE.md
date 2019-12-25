# Development notes

## Test

Do `pytest` under root folder.

```sh
pytest
```

## Package and upload to PyPi

Package as follows.

```sh
python setup.py sdist bdist_wheel
```

Then upload.

```sh
twine upload dist/dl_cliche-0.1.2-py3-none-any.whl
```

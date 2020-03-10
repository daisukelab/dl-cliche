#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name='dl-cliche',
    version='0.1.3',
    description='dl-cliche: Packaging cliche utilities',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='daisukelab',
    author_email='contact.daisukelab@gmail.com',
    install_requires=['numpy', 'pandas', 'matplotlib', 'tqdm',
                      'easydict', 'imbalanced-learn', 'tables', 'openpyxl'],
    url='https://github.com/daisukelab/dl-cliche',
    license=license,
    packages=find_packages(exclude=('test', 'docs')),
    test_suite='test',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)

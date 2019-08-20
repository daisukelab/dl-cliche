#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='dl-cliche',
    version='0.0.2',
    description='A small python library to summarize all cliche codes',
    #long_description=readme,
    author='daisukelab',
    author_email='foo@bar.com',
    install_requires=['numpy'],
    url='https://github.com/daisukelab/dl-cliche',
    license=license,
    packages=find_packages(exclude=('test', 'docs')),
    test_suite='test'
)

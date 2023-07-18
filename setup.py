#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.md').read()
doclink = """
## Documentation

The full documentation is at http://simanneal.rtfd.org."""
history = open('HISTORY.md').read().replace('.. :changelog:', '')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='simanneal',
    version='0.1.0',
    description='an updated version of perrygeo/simanneal',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    long_description_content_type="test/markdown",
    author='Joe Yesselman',
    author_email='jyesselm@unl.edu',
    url='https://github.com/jyesselm/simanneal',
    packages=[
        'simanneal',
    ],
    package_dir={'simanneal': 'simanneal'},
    py_modules=[
        'simanneal/anneal',
        'simanneal/logger'
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='simanneal',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    entry_points = {
        'console_scripts' : [
        ]
    }
)

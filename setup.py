"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requires = ["requests>=2.14.2"]


setup(
    name='MF_intern',
    version='0.0.1',
    description='movie recommendation program for intern test',
    url='',
    author='MSA8D8',  # Optional
    author_email='',  # Optional

    license='MIT',
    keywords='sample setuptools development',
    packages=[
        "your_package",
        "your_package.subpackage",
    ],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)

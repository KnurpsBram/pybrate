import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="fvrt_audio",
    version="1.0",
    description='My Favourite Audio Tools Implemented in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bram Kooiman',
    author_email='bramkooiman94@gmail.com',
    packages=['fvrt_audio'],
    install_requires=[
        'numpy',
        'torch',
    ],
    include_package_data=True
)

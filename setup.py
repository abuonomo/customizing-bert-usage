from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='dsbert',
    version='0.1.0',
    description='Template for using Google\'s BERT for binary classification',
    long_description=readme,
    author='Anthony Buonomo',
    author_email='anthony.r.buonomo@nasa.gov',
    packages=find_packages(where='./src')
)
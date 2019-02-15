from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='dsbert',
    version='0.1.0',
    description='Template for using Google\'s BERT for binary classification',
    long_description=readme,
    author='Anthony Buonomo',
    author_email='anthony.r.buonomo@nasa.gov',
    packages=find_packages(where='./src'),
    install_requires=[
       'PyYAML>=3.13',
       'tensorflow>=1.11.0',
       'pandas>=0.23.4',
       'numpy>=1.15.2',
       'Flask>=1.0.2',
    ],
)
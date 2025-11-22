from setuptools import setup, find_packages

setup(
    name = "polynomial activations",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=2.3.4',
    ],
)
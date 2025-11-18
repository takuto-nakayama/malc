from setuptools import setup, find_packages

setup(
    name='package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'transformers'
    ],
)

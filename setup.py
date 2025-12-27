from setuptools import setup, find_packages

setup(
    name='packages for MaLC',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'numpy',
        'pandas',
		'persim',
		'ripser',
        'scikit-learn',
        'scipy',
		'seaborn',
        'stanza'
    ]
)

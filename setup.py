from setuptools import setup, find_packages

setup(
    name='betaburst',
    version='0.1.0',
    description='Beta burst analysis toolkit',
    author='Ludovic Darmet',
    author_email='ludovic.darmet@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mne',
        'pandas',
        'scikit-learn',
        'fooof'
    ],
    entry_points={
        'console_scripts': [
            'betaburst = betaburst.__main__:main'
        ]
    }
)
#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='PeakBotMRM',
    version='0.9.0',
    author='Christoph Bueschl',
    author_email='christoph.bueschl [the little email symbol] univie.ac.at',
    packages=find_packages(),
    url='https://github.com/christophuv/PeakBot',
    license='LICENSE',
    description='A machine-learning CNN model for peak picking in MRM EICs',
    long_description=open('README.md').read(),
    install_requires=[
        ## main libraries for machine learning related tasks
        "tensorflow >= 2.8.0",
        "tensorflow_addons >= 0.16.1",
        "tensorflow_probability >= 0.16.0",

        ## other main function libraries
        # "numpy",  ## removed as a dependency of tensorflow 
        "pandas",
        "matplotlib",
        "plotnine",
        "portalocker", 

        ## handling of LC-MS data
        "pymzml", 

        ## other libraries
        "tqdm",
        "natsort",    
        "py-cpuinfo",
        "psutil"
    ],
)

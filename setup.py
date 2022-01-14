#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='PeakBotMRM',
    version='0.4.6',
    author='Christoph Bueschl',
    author_email='christoph.bueschl [the little email symbol] univie.ac.at',
    packages=find_packages(),#['peakbot', 'peakbot.train'],
    url='https://github.com/christophuv/PeakBot_MRM',
    license='LICENSE',
    description='A machine-learning CNN model for peak picking in MRM EICs',
    long_description=open('README.md').read(),
    install_requires=[
        ## main libraries for machine learning related tasks
        "tensorflow == 2.5.0",
        "tensorflow_addons == 0.15.0",

        ## other main function libraries
        "numba == 0.53.1",
        "pandas == 1.2.3",
        "matplotlib >= 3.4.2",
        "plotnine >= 0.8.0",

        ## handling of LC-MS data
        "pymzml == 2.5.0", 

        ## other libraries
        "tqdm >= 4.61.2",
        "natsort >= 7.1.1",    
        "py-cpuinfo >= 8.0.0",
        "psutil == 5.9.0"
    ],
)

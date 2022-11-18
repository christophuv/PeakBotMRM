# PeakBot for MRM
This is an adaption of [PeakBot](https://github.com/christophuv/PeakBot) for MRM datasets. 



## Install PeakBotMRM for raw data processing with an existing model
PeakBotMRM is a pyhton package and can thus be run on different operating system. However, the recommended method for installation is to run it in a virtual environment with Anaconda as all CUDA dependencies can automatically be installed there. 

### GPU support
PeakBotMRM uses the graphics processing unit of the PC for computational intensive tasks such as the generation of the large training dataset or the training of the CNN model. Thus, it requires a CUDA-enabled graphics card from Nvidia as well as the CUDA tookit and the cuDNN libraries to be installed. For further information about these packages please consult the official documentation of Nvidia at https://developer.nvidia.com/cuda-downloads, https://developer.nvidia.com/cudnn and https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html. 
If the Anaconda environment together with a virtual environment is used, these steps can be omitted as they can easily be installed there.

Note: Different GPUs have a different number of streaming-processors. Thus, the blockdim and griddim need to be chosen accordingly. Please adapt these values to your GPU. To find good values for your particular GPU, the script quickFindCUDAParameters.py from the PeakBotMRM examples repository can be used. It iterates over possible combinations of the parameters and tries to detect peaks with these. When all have been tested, a list with the best is presented and these parameters should then be used.
Note: If an exportBatchSize of 2048 requires some 4GB of GPU-memory. If you have less, try reducing this value to 1024 of 512. 

### Windows 10
0. Optional (with GPU): Update your Nvidia GPU driver to the latest available. Driveras can be downloaded from https://www.nvidia.com/Download/index.aspx. 

0. Optional (with GPU): Increase the WDMM TDR (timeout for GPU refresh). This can be done via the Registry Editor. For this, plese press the Windows key and enter regedit. Do not start it, but right-click on the entry "Registry Editor" and start it as an administrator by clicking "Run as Administrator". Enter the credentials of your administrator account and start the Registry Editor. Navigate to the directory "HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers" and create the key "TdrLevel" as a REG_DWORD with the value 3. Add another key "TdrDelay" with the value 45. This will increase the timeout for the display refresh to 45 seconds. Then restart your system. More information about these to keys can be found at https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/ and https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918(v=vs.85).aspx.

0. Optional: Download and install the ProteoWizard toolbox for converting to the mzML file format. Do not change the path of the installation folder during installation as this might cause PeakBotMRM to not find the executable automatically. Available at https://proteowizard-teamcity-artifacts.s3.us-west-2.amazonaws.com/ProteoWizard/bt83/1883839/pwiz-setup-3.0.22187.a58d608-x86_64.msi

1. Download and install Anaconda from https://www.anaconda.com. 

2. Start the "Anaconda prompt" from the start menu and navigate to any folder where you would like to setup PeakBotMRM (e.g., C:/PeakBotMRM)

3. Create a new conda virtual environment and activate it with the following commands:
```
    ## Without CUDA graphics card
    curl https://raw.githubusercontent.com/christophuv/PeakBotMRM/main/AnacondaEnvironment/CEMM_Windows10_python310.yml > CEMM_Windows10_python310.yml
    conda env create --file CEMM_Windows10_python310.yml
```

4. Activate the new environment:
```
    conda activate PeakBotMRM
```

5. Install the PeakBotMRM framework with the command:
```
    git clone https://github.com/christophuv/PeakBotMRM.git
```

6. Download a pre-trained model (optional, only if available to your organization):
```
    cd PeakBotMRM
    mkdir models
    cd models
    curl http://cemmgit.int.cemm.at/cbueschl/peakbotmrm_models/-/raw/master/METAB02__0a967796629c438387f2ba81482cd37e/METAB02__0a967796629c438387f2ba81482cd37e.h5?inline=false
    cd ../..
```

7. Run PeakBotMRM:
```
    python ./PeakBotMRM/src/gui.py
```


## Train a new model for a custom LC-QQQ-MS method

### Windows 10

0. Optional Generate a new conda environment with the steps 0 - 5 in the section 'Install PeakBotMRM for raw data processing with an existing model'. 
1. Start the "Anaconda prompt" from the start menu and navigate to any folder where you would like to start the training. 
2. Activate the PeakBotMRM conda environment
```
    conda activate PeakBotMRM
```

3. Download the PeakBotMRM repository to this folder
```
mkdir PeakBotMRM
cd PeakBotMRM
git clone https://github.com/christophuv/PeakBotMRM.git
cd ..
```

5. Create a new python file and implement the settings of the training and validation dataset(s)
```
################################
### Settings

import os
## Experiment name
expName = "Train_METAB02"
## Experiment directory
expDir = os.path.join(".", expName)
## Modelfiles
modelPath = os.path.join(expDir, "tmp")
## Logging directory
logDir = os.path.join(expDir, "logs")
## History object
historyFile = os.path.join(expDir, "History_Training.pandas.pickle")
### Training datasets
trainDSs = []
### Validation datasets (independent from training dataset)
valDSs = []

## R100140_METABO02 dataset
trainDSs.append(
    {   ## The dataset's name
        "DSName"              : "Ref_R100140",
        ## File where the MRM information about the substances is saved
        "transitions"         : "./Reference/transitions_UntilR100269.tsv",
        ## File with the manual integration results
        "GTPeaks"             : "./Reference/R100140_Integrations.csv",
        ## Path to the chromatograms
        "samplesPath"         : "./Reference/R100140_METAB02_MCC025_20200306",
        ## Substances to exlcude (None refers to no substance)
        "excludeSubstances"   : ["Hexose 1-phosphate", "Hexose 6-phosphate"], 
        ## Substances to use (None refers to all substances)
        "includeSubstances"   : None,
        ## Samples to be used or not (None refers to al substances)
        "sampleUseFunction"   : None,
        ## (1, function) .. check peak attributes with function, None or (0, anything) .. dont check peak attributes, (-1, function) .. check peak attributes with function but use the inverse
        "checkPeakAttributes" : None
    })





################################
### Imports

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = 0    ## TODO set number of CUDA devices

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    ## taken from https://stackoverflow.com/a/61010643
    ## Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("Created virtual GPU device with a memory limitation")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
tf.get_logger().setLevel('WARNING')


## Import PeakBotMRM from directory not as installed package (can be omitted if the package is installed)
import sys
sys.path.append(os.path.join(".", "PeakBotMRM", "src")) 
## Load the PeakBotMRM package
import PeakBotMRM.train





################################
### Training

PeakBotMRM.train.trainPeakBotMRMModel(expName, 
                                      trainDSs, valDSs, 
                                      modelFile = os.path.join(modelPath, "PBMRM.h5", 
                                      expDir = expDir, logDir = logDir, 
                                      showPeakMetrics = False,
                                      balanceDataset = False, balanceAugmentations = True, 
                                      addRandomNoise = True, 
                                      shiftRTs = True, 
                                      aug_augment = True, aug_addAugPeaks = True, aug_maxAugPeaksN = 3,
                                      stratifyDataset = True, intThres = None, peakWidthThres = 0.3, 
                                      useEachInstanceNTimes = 5,
                                      historyFile = historyFile,
                                      comment="") 

PeakBotMRM.train.createHistory(historyFile, locationAndPrefix = os.path.join(expDir, "summary_"))
```

6. Optionally: Modify the GPU index and/or the memory limitation
7. Play around with the different parameters of the trainPeakBotMRMModel function




## Acknowledgements

Figures and illustrations have been designed using resources from https://flaticon.com. 
# PeakBot for MRM

This is an adaption of [PeakBot](https://github.com/christophuv/PeakBot) for MRM datasets. 



## Install PeakBox
PeakBotMRM is a pyhton package and can thus be run on different operating system. However, the recommended method for installation is to run it in a virtual environment with Anaconda as all CUDA dependencies can automatically be installed there. 

### GPU support
PeakBotMRM uses the graphics processing unit of the PC for computational intensive tasks such as the generation of the large training dataset or the training of the CNN model. Thus, it requires a CUDA-enabled graphics card from Nvidia as well as the CUDA tookit and the cuDNN libraries to be installed. For further information about these packages please consult the official documentation of Nvidia at https://developer.nvidia.com/cuda-downloads, https://developer.nvidia.com/cudnn and https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html. 
If the Anaconda environment together with a virtual environment is used, these steps can be omitted as they can easily be installed there.

Note: Different GPUs have a different number of streaming-processors. Thus, the blockdim and griddim need to be chosen accordingly. Please adapt these values to your GPU. To find good values for your particular GPU, the script quickFindCUDAParameters.py from the PeakBotMRM examples repository can be used. It iterates over possible combinations of the parameters and tries to detect peaks with these. When all have been tested, a list with the best is presented and these parameters should then be used.
Note: If an exportBatchSize of 2048 requires some 4GB of GPU-memory. If you have less, try reducing this value to 1024 of 512. 

### Windows 10
0. Optional (with GPU): Update your Nvidia GPU driver to the latest available. Driveras can be downloaded from https://www.nvidia.com/Download/index.aspx. 

0. Optional (with GPU): Increase the WDMM TDR (timeout for GPU refresh). This can be done via the Registry Editor. For this, plese press the Windows key and enter regedit. Do not start it, but right-click on the entry "Registry Editor" and start it as an administrator by clicking "Run as Administrator". Enter the credentials of your administrator account and start the Registry Editor. Navigate to the directory "HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers" and create the key "TdrLevel" as a REG_DWORD with the value 3. Add another key "TdrDelay" with the value 45. This will increase the timeout for the display refresh to 45 seconds. Then restart your system. More information about these to keys can be found at https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/ and https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918(v=vs.85).aspx.

0. Optional: Download and install the ProteoWizard toolbox for converting to the mzML file format. Available at https://proteowizard-teamcity-artifacts.s3.us-west-2.amazonaws.com/ProteoWizard/bt83/1883839/pwiz-setup-3.0.22187.a58d608-x86_64.msi

1. Download and install Anaconda from https://www.anaconda.com. 

2. Start the "Anaconda prompt" from the star menu. 

3. Create a new conda virtual environment and activate it with the one of the following commands depending whether or not you have a CUDA-enabled graphics card:

```
    ## Without CUDA graphics card
    curl http://cemmgit.int.cemm.at/cbueschl/PeakBotMRM/-/raw/main/AnacondaEnvironment/CEMM_Windows10_python38_withGPU.yml?inline=false > CEMM_Windows10_python38_withGPU.yml
    conda env create --file CEMM_Windows10_python38_withGPU.yml
```

4. Activate the new environment:

```
    conda activate PeakBotMRM
```

5. Install the PeakBotMRM framework with the command:

```
    git clone http://cemmgit.int.cemm.at/cbueschl/PeakBotMRM.git
```

6. Download a pre-trained model:

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



## Acknowledgements

Figures and illustrations have been designed using resources from https://flaticon.com. 
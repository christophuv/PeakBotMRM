rem ############################
rem ##
rem ## PeakBotMRM call script
rem ##
rem ############################

rem only if necessary
rem install miniconda from https://docs.conda.io/en/latest/miniconda.html

rem only if necessary
rem conda env create -f ..\PeakBotMRM\AnacondaEnvironment\CEMM_Windows10_python38_withGPU.yml

rem activate conda PeakBotMRM environment
call "C:\ProgramData\Miniconda3\Scripts\activate" PeakBotMRM

rem start GUI of PeakBotMRM
python C:\Projects\PeakBot_MRM\PeakBotMRM\src\gui.py

set /P id=Press ENTER to exit 

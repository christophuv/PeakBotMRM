#
## Manual tasks
## ------------
## Execute these manual tasks before invoking the packing script
##
## modify C:\Users\cbueschl\.conda\envs\PeakBotMRM\Lib\site-packages\pyqtgraph\graphicsItems\PlotItem\PlotItem.py
## cange the following line
## # f'.plotConfigTemplate_{QT_LIB.lower()}', package=__package__)
## f'.plotConfigTemplate_{QT_LIB.lower()}', package='pyqtgraph.graphicsItems.PlotItem')
##  
## modify C:\Users\cbueschl\.conda\envs\PeakBotMRM\Lib\site-packages\pyqtgraph\imageview\ImageView.py
## cange the following line
## ## f'.ImageViewTemplate_{QT_LIB.lower()}', package=__package__)
## f'.ImageViewTemplate_{QT_LIB.lower()}', package='pyqtgraph.imageview')
##  
## modify C:\Users\cbueschl\.conda\envs\PeakBotMRM\Lib\site-packages\pyqtgraph\graphicsItems\ViewBox\ViewBoxMenu.py
## cange the following line
## ## f'.axisCtrlTemplate_{QT_LIB.lower()}', package=__package__)
## f'.axisCtrlTemplate_{QT_LIB.lower()}', package='pyqtgraph.graphicsItems.ViewBox')
##
## modify C:\Users\cbueschl\.conda\envs\PeakBotMRM\Lib\site-packages\pyqtgraph\__init__.py
## comment out these two imports
## #from .metaarray import MetaArray
## #from .ordereddict import OrderedDict
##
## modify C:\Users\cbueschl\.conda\envs\PeakBotMRM\Lib\site-packages\palettable\colorbrewer\colorbrewer.py
## modify 
## f = resource_string(__name__, 'data/colorbrewer_all_schemes.json')
## to
##    import sys
##    if hasattr(sys, 'frozen'):
##    # retrieve path from sys.executable
##    import os
##    rootdir = os.path.abspath(os.path.dirname(sys.executable))
##    with open(os.path.join(rootdir, 'palettable', 'colorbrewer', 'data', 'colorbrewer_all_schemes.json'), "r") as fin:
##        f = fin.read()
##    else:
##    # assign a value from __file__
##    f = resource_string(__name__, 'data/colorbrewer_all_schemes.json').decode('ascii')
##
## modify C:\Users\cbueschl\.conda\envs\PeakBotMRM\Lib\site-packages\sklearn\_distributor_init.py
##    import sys
##    if hasattr(sys, 'frozen'):
##      # retrieve path from sys.executable
##      import os
##      libs_path = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "sklearn", ".libs"))
##      
##    else:
##        libs_path = op.join(op.dirname(__file__), ".libs")
##
# 
 
 
from distutils.core import setup
#import py2exe
from cx_Freeze  import setup, Executable

from glob import glob
import os
import sys
import shutil

basePath = "C:/Users/cbueschl/.conda/envs/PeakBotMRM/"

data_files = [("Microsoft.VC90.CRT", glob('C:/Program Files/Microsoft Visual Studio 9.0/VC/redist/x86/Microsoft.VC90.CRT/*.*')),
              ("imageformats", glob(basePath + "Lib/site-packages/PyQt6/Qt6/plugins/imageformats/*.*")),
              ("PyQt6", glob(basePath + "Lib/site-packages/PyQt6/*.pyd")),
              (".", [basePath + "Lib/site-packages/llvmlite/binding/llvmlite.dll",
                     basePath + "Lib/site-packages/PyQt6/Qt6/bin/Qt6Widgets.dll",
                     basePath + "Lib/site-packages/PyQt6/Qt6/bin/Qt6Core.dll",
                     basePath + "Lib/site-packages/PyQt6/Qt6/bin/Qt6Gui.dll"]),
              ("sklearn/.libs", glob(basePath + "Lib/site-packages/sklearn/.libs/*.dll")),
              ]

includes = [
            "scipy", "scipy.special.cython_special",
            "statsmodels", "statsmodels.tsa.statespace._filters", "statsmodels.tsa.statespace._filters._conventional", 
            "statsmodels.tsa.statespace._kalman_filter", "statsmodels.tsa.statespace._filters._univariate",
            "statsmodels.tsa.statespace._filters._univariate_diffuse", "statsmodels.tsa.statespace._filters._inversions",
            "statsmodels.tsa.statespace._smoothers._alternative", "statsmodels.tsa.statespace._smoothers._classical", 
            "statsmodels.tsa.statespace._smoothers._conventional", "statsmodels.tsa.statespace._smoothers._univariate", 
            "statsmodels.tsa.statespace._smoothers._univariate_diffuse", 
            "sklearn", "scipy.sparse.csgraph._validation", "sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes", "sklearn.utils._cython_blas", 
            "sklearn.utils._weight_vector", 
            "tensorflow", "tensorflow_addons", "tensorflow_probability"
            ]


include_files = []
include_files.append((basePath + "Lib/site-packages/palettable/colorbrewer/data/colorbrewer_all_schemes.json", "palettable/colorbrewer/data"))
for g in glob(basePath + "Lib/site-packages/sklearn/.libs/*.dll"):
    include_files.append((g, "sklearn/.libs"))
for g in glob("./gui-resources/*"):
    include_files.append((g, "gui-resources"))

build_exe_options = {"packages": includes, "excludes": []}

setup(name = "PeakBotMRM",
    executables = [Executable("./gui.py")], 
    options = {"build_exe": build_exe_options},
    data_files = data_files)

os.rename("./build/exe.win-amd64-3.8/gui.exe", "./build/exe.win-amd64-3.8/PeakBotMRM_GUI.exe")

for s, t in include_files:
    n = os.path.basename(os.path.join("C:/Users/cbueschl/.conda/envs/PeakBotMRM/Lib/site-packages", s))
    os.makedirs(os.path.join("./build/exe.win-amd64-3.8", t), exist_ok=True)
    shutil.copyfile(s, os.path.join("./build/exe.win-amd64-3.8", t, n))


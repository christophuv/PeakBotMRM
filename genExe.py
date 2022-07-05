from distutils.core import setup
import py2exe

from glob import glob


import os
import sys
sys.path.append(os.path.join("..", "PeakBotMRM", "src"))


data_files = [("Microsoft.VC90.CRT", glob('C:\\Program Files\\Microsoft Visual Studio 9.0\\VC\redist\\x86\\Microsoft.VC90.CRT\\*.*'))]
includes = ["sip", "PyQt6", "PyQt6.QtCore", "PyQt6.QtOpenGL", "PyQt6.QtGui"]

setup(name = "PeakBotMRM",
      windows = [{"script": "./src/PeakBotMRM/gui/gui.py"}], 
      options = {"py2exe": {"includes": includes}},
      data_files = data_files)

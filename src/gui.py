import logging
logging.root.setLevel(logging.NOTSET)
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.WARNING)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler("PeakBotMRM_run.log")
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

import os
import sys
pythonScriptDirectory = None
if hasattr(sys, 'frozen'):
    # retrieve path from sys.executable
    pythonScriptDirectory = os.path.abspath(os.path.dirname(sys.executable))
else:
    # assign a value from __file__
    pythonScriptDirectory = os.path.abspath(os.path.dirname(__file__))
import os
import PySimpleGUI as sg
try:
    splashImage = os.path.join(pythonScriptDirectory, "gui-resources", "robot_loading.png")
    window = sg.Window("", [[sg.Image(splashImage)]], transparent_color=sg.theme_background_color(), no_titlebar=True, keep_on_top=True, finalize=True)
    window.bring_to_front()
except:
    logging.warning("Cannot show splash screen")
    
from typing import OrderedDict
import functools
import shutil
import sys
import natsort
import re
import pickle
import subprocess
from pathlib import Path
import copy
import csv
import datetime
import platform
import regex

import unit_converter
from unit_converter.converter import convert, converts
unit_converter.data.UNITS["L"] = unit_converter.units.Unit("L", "liter", L=1)
## converts("1 g * L^-1", "µg * L^-1")

#from rdkit import Chem
#from rdkit.Chem import Draw

import PyQt6.QtWidgets
import PyQt6.QtCore
import PyQt6.QtGui

import pyqtgraph
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')
pyqtgraph.setConfigOptions(antialias=True)
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy as np
import pacmap

import pandas as pd
import plotnine as p9
import tempfile
import base64


## Specific tensorflow configuration. Can re omitted or adapted to users hardware
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import PeakBotMRM
import PeakBotMRM.predict

try:
    window.close()
except:
    pass



dpi = 72
size_inches = (5, 2)                                        # size in inches (for the plot)
size_px = int(size_inches[0]*dpi), int(size_inches[1]*dpi)  # For the canvas
fontSizeP9 = 8



def sortSamples(sampleNames, importantSamplesRegEx):
    order = []
    for imp in importantSamplesRegEx:
        for samp in natsort.natsorted(sampleNames, key = lambda x: x.lower()):
            if re.search(imp, samp) and samp not in order:
                order.append(samp)
    
    for samp in natsort.natsorted(sampleNames, key = lambda x: x.lower()):
        if samp not in order:
            order.append(samp)
    
    return order




class Experiment:
    def __init__(self, expName = "", substances = None, integrations = None, sampleInfo = None):
        self.expName = expName
        self.substances = substances
        self.integrations = integrations
        self.sampleInfo = sampleInfo

    def saveToFile(self, toFile, additionalData = None):
        tempIntegrations = {}
        for sub in self.integrations:
            tempIntegrations[sub] = {}
            for samp in self.integrations[sub]:
                inte = self.integrations[sub][samp]
                
                inte2 = PeakBotMRM.Integration(inte.foundPeak, inte.rtStart, inte.rtEnd, inte.area, inte.chromatogram)
                inte2.type = inte.type
                inte2.comment = inte.comment
                inte2.foundPeak = inte.foundPeak
                inte2.rtStart = inte.rtStart
                inte2.rtEnd = inte.rtEnd
                inte2.area = inte.area
                inte2.istdRatio = inte.istdRatio
                inte2.concentration = inte.concentration
                inte2.chromatogram = inte.chromatogram
                inte2.other = {}
                
                tempIntegrations[sub][samp] = inte2
                
        try:
            with open(toFile, "wb") as fout:
                pickle.dump((copy.deepcopy(self.expName), copy.deepcopy(self.substances), tempIntegrations, copy.deepcopy(self.sampleInfo), additionalData), fout)
                return True
        except Exception as ex:
            logging.exception("Exception during binary experiment output")
            return False




## Copied from https://stackoverflow.com/a/61389578
class QHSeperationLine(PyQt6.QtWidgets.QFrame):
  '''
  a horizontal seperation line\n
  '''
  def __init__(self, fixedHeight = 3):
    super().__init__()
    self.setMinimumWidth(1)
    self.setFixedHeight(fixedHeight)
    self.setFrameShape(PyQt6.QtWidgets.QFrame.Shape.HLine)
    self.setFrameShadow(PyQt6.QtWidgets.QFrame.Shadow.Plain)
    return


class WebViewDialog(PyQt6.QtWidgets.QDialog):
    def __init__(self, parent=None, title = "", html = "<h>No content</h>", url = None):
        super(WebViewDialog, self).__init__(parent)
        self.setWindowTitle(title)
             
        ## TODO this does not work. The import disables any file dialogs. Fix the bug    
        import PyQt6.QtWebEngineWidgets
        
        grid = PyQt6.QtWidgets.QGridLayout()
        webView = PyQt6.QtWebEngineWidgets.QWebEngineView()
        if url is not None:
            webView.setUrl(url)
        else:
            webView.setHtml(html)
        grid.addWidget(webView, 0, 0)
        self.setLayout(grid)

class TimerMessageBox(PyQt6.QtWidgets.QMessageBox):
    def __init__(self, parent=None, title = "", text = "", timeout = 3):
        super(TimerMessageBox, self).__init__(parent)
        self.setWindowTitle(title + " - automatically closing")
        self.setModal(True)
        self.time_to_wait = timeout
        self.text = text
        self.setText(self.text + "<br><br>// closing automatically in {0} secondes.".format(timeout))
        self.setStandardButtons(PyQt6.QtWidgets.QMessageBox.StandardButton.Ok)
        self.timer = PyQt6.QtCore.QTimer(self)
        self.timer.setInterval(1001)
        self.timer.timeout.connect(self.changeContent)
        self.timer.start()
        self.exec()

    def changeContent(self):
        self.setText(self.text + "<br><br>// closing automatically in {0} secondes.".format(self.time_to_wait))
        if self.time_to_wait <= 0:
            self.close()
            self.hide()
        self.time_to_wait -= 1

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

        
        
        


class EditTableDialog(PyQt6.QtWidgets.QDialog):
    def __init__(self, parent=None, data = None):
        super(EditTableDialog, self).__init__(parent)
        
        self.setWindowTitle("Edit data")
        
        rowi = 0
        grid = PyQt6.QtWidgets.QGridLayout()
        grid.addWidget(PyQt6.QtWidgets.QLabel("<b>Edit table data</b>"), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        self.table = pyqtgraph.TableWidget(editable = True, sortable = True)
        self.table.setData(data)
        ## TODO self.table.setCellWidget(3,3, PyQt6.QtWidgets.QDoubleSpinBox())
        grid.addWidget(self.table, rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        self.addCol = PyQt6.QtWidgets.QPushButton("Add column")
        self.addCol.clicked.connect(self.addColumn)
        grid.addWidget(self.addCol, rowi, 0, 1, 1)
        
        self.resetCells = PyQt6.QtWidgets.QPushButton("Reset selected cells")
        self.resetCells.clicked.connect(self.resetColumnValues)
        grid.addWidget(self.resetCells, rowi, 1, 1, 1)
        
        self.fillCells = PyQt6.QtWidgets.QPushButton("Fill with first cells")
        self.fillCells.clicked.connect(self.copyCellValues)
        grid.addWidget(self.fillCells, rowi, 2, 1, 1)
        
        rowi = rowi + 1
        self.rmCol = PyQt6.QtWidgets.QPushButton("Remove column")
        self.rmCol.clicked.connect(self.removeColum)
        self.rmCol.setVisible(False)
        grid.addWidget(self.rmCol, rowi, 1, 1, 1)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
                
        rowi = rowi + 1
        self.cancel = PyQt6.QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.reject)
        grid.addWidget(self.cancel, rowi, 1, 1, 1)
        
        self.fin = PyQt6.QtWidgets.QPushButton("Ok")
        self.fin.clicked.connect(self.accept)
        grid.addWidget(self.fin, rowi, 2, 1, 1)
        
        self.setLayout(grid)
        
    def copyCellValues(self):
        sel = self.table.selectedItems()
        colVal2Copy = {}
        if sel is not None:
            for selCell in sel:
                if selCell.column() not in colVal2Copy:
                    colVal2Copy[selCell.column()] = selCell.text()
                else:
                    selCell.setText(colVal2Copy[selCell.column()])
    
    def resetColumnValues(self):
        sel = self.table.selectedItems()
        if sel is not None:
            for selCell in sel:
                selCell.setText("")
    
    def addColumn(self):
        name = PyQt6.QtWidgets.QInputDialog.getText(self, "Add column", "please specify the new column's name (only text allowed)")
        
        if name[1] and name[0] is not None and name[0] != "":
            self.table.insertColumn(self.table.columnCount())
            self.table.setHorizontalHeaderItem(self.table.columnCount() - 1, PyQt6.QtWidgets.QTableWidgetItem(name[0]))
    
    def removeColum(self):
        print("TODO implement")
        
    def getUserData(self):
        data = []
        headers = set()
        for rowi in range(self.table.rowCount()):
            temp = {}
            for coli in range(self.table.columnCount()):
                colname = self.table.horizontalHeaderItem(coli)
                if colname is not None:
                    colname = colname.text()
                    headers.add(colname)
                    item = self.table.item(rowi, coli)
                    if item is not None:
                        temp[colname] = self.table.item(rowi, coli).text()
                    else:
                        temp[colname] = None
            data.append(temp)
        
        return headers, data
    



class TrainModelDialog(PyQt6.QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TrainModelDialog, self).__init__(parent)
        
        self.setWindowTitle("Train new model - parameters")
        
        rowi = 0
        grid = PyQt6.QtWidgets.QGridLayout()
        grid.addWidget(PyQt6.QtWidgets.QLabel("<b>Train a new model</b>"), rowi, 0)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Model name"), rowi, 0)
        self.modelName = PyQt6.QtWidgets.QLineEdit("")
        grid.addWidget(self.modelName, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Comment"), rowi, 0)
        self.comment = PyQt6.QtWidgets.QLineEdit("")
        grid.addWidget(self.comment, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Folder to save model and log"), rowi, 0)
        self.loadFolder = PyQt6.QtWidgets.QPushButton("Open")
        self.folderPath = PyQt6.QtWidgets.QLabel("")
        self.loadFolder.clicked.connect(self.openRaw)
        grid.addWidget(self.loadFolder, rowi, 1, 1, 1)
        grid.addWidget(self.folderPath, rowi, 2, 1, 1)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Add random noise"), rowi, 0)
        self.addRandomNoise = PyQt6.QtWidgets.QCheckBox("")
        self.addRandomNoise.setChecked(True)
        grid.addWidget(self.addRandomNoise, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Randomization factor (maximum)"), rowi, 0)
        self.maxRandFactor = PyQt6.QtWidgets.QDoubleSpinBox()
        self.maxRandFactor.setMinimum(0.01)
        self.maxRandFactor.setMaximum(0.5)
        self.maxRandFactor.setValue(0.1)
        grid.addWidget(self.maxRandFactor, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Noise level add (maximum)"), rowi, 0)
        self.maxNoiseLevelAdd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.maxNoiseLevelAdd.setMinimum(0.01)
        self.maxNoiseLevelAdd.setMaximum(0.5)
        self.maxNoiseLevelAdd.setValue(0.1)
        grid.addWidget(self.maxNoiseLevelAdd, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Shift RTs"), rowi, 0)
        self.shiftRTs = PyQt6.QtWidgets.QCheckBox("")
        self.shiftRTs.setChecked(True)
        grid.addWidget(self.shiftRTs, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Rt shift (maximum)"), rowi, 0)
        self.maxShift = PyQt6.QtWidgets.QDoubleSpinBox()
        self.maxShift.setMinimum(0.01)
        self.maxShift.setMaximum(10)
        self.maxShift.setValue(0.15)
        grid.addWidget(self.maxShift, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Balance augmentations"), rowi, 0)
        self.balanceAugmentations = PyQt6.QtWidgets.QCheckBox("")
        self.balanceAugmentations.setChecked(True)
        grid.addWidget(self.balanceAugmentations, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Number of derived instances from each instance"), rowi, 0)
        self.nInstances = PyQt6.QtWidgets.QSpinBox()
        self.nInstances.setMinimum(1)
        self.nInstances.setMaximum(100)
        self.nInstances.setValue(8)
        grid.addWidget(self.nInstances, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Plot results"), rowi, 0)
        self.plotResults = PyQt6.QtWidgets.QCheckBox("")
        self.plotResults.setChecked(True)
        grid.addWidget(self.plotResults, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("TODO: add augmentation options and derive default values automatically"), rowi, 0)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        self.fin = PyQt6.QtWidgets.QPushButton("Start training")
        self.fin.clicked.connect(self.accept)
        grid.addWidget(self.fin, rowi, 2, 1, 1)
        
        self.cancel = PyQt6.QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.reject)
        grid.addWidget(self.cancel, rowi, 1, 1, 1)
        
        self.setLayout(grid)
        
    def openRaw(self):
        fDir = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder for model and log", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if fDir:
            self.folderPath.setText(fDir)
    
    def getUserData(self):
        return (self.modelName.text(), self.traPath.text(), self.folderPath.text(), self.resPath.text())
    


class OpenExperimentDialog(PyQt6.QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(OpenExperimentDialog, self).__init__(parent)
        
        self.setWindowTitle("New experiment")
        
        rowi = 0
        grid = PyQt6.QtWidgets.QGridLayout()
        grid.addWidget(PyQt6.QtWidgets.QLabel("<b>Load experiment</b>"), rowi, 0)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Experiment name"), rowi, 0)
        self.expName = PyQt6.QtWidgets.QLineEdit("")
        grid.addWidget(self.expName, rowi, 1, 1, 2)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Transitions file"), rowi, 0)
        self.loadTra = PyQt6.QtWidgets.QPushButton("Open")
        self.traPath = PyQt6.QtWidgets.QLabel("")
        self.loadTra.clicked.connect(self.openTransitions)
        grid.addWidget(self.loadTra, rowi, 1, 1, 1)
        grid.addWidget(self.traPath, rowi, 2, 1, 1)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Raw LCMS data"), rowi, 0)
        self.loadRaw = PyQt6.QtWidgets.QPushButton("Open")
        self.rawPath = PyQt6.QtWidgets.QLabel("")
        self.loadRaw.clicked.connect(self.openRaw)
        grid.addWidget(self.loadRaw, rowi, 1, 1, 1)
        grid.addWidget(self.rawPath, rowi, 2, 1, 1)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Processed results"), rowi, 0)
        self.loadRes = PyQt6.QtWidgets.QPushButton("Open")
        self.resPath = PyQt6.QtWidgets.QLabel("")
        self.loadRes.clicked.connect(self.openResults)
        grid.addWidget(self.loadRes, rowi, 1, 1, 1)
        grid.addWidget(self.resPath, rowi, 2, 1, 1)
        
        rowi = rowi + 1
        grid.addWidget(PyQt6.QtWidgets.QLabel("Delimiter character"), rowi, 0)
        self.delim = PyQt6.QtWidgets.QComboBox()
        self.delim.addItems(["TAB", ",", ";", "$"])
        grid.addWidget(self.delim, rowi, 1, 1, 1)
        
        rowi = rowi + 1
        grid.addWidget(QHSeperationLine(), rowi, 0, 1, 3)
        
        rowi = rowi + 1
        self.fin = PyQt6.QtWidgets.QPushButton("Open experiment")
        self.fin.clicked.connect(self.accept)
        grid.addWidget(self.fin, rowi, 2, 1, 1)
        
        self.cancel = PyQt6.QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.reject)
        grid.addWidget(self.cancel, rowi, 1, 1, 1)
        
        self.setLayout(grid)
        
    def openTransitions(self):
        fName = PyQt6.QtWidgets.QFileDialog.getOpenFileName(self, "Open transitions file", filter="Tab separated values files (*.tsv);;Comma separated values files (*.csv);;All files (*.*)", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if fName[0]:
            self.traPath.setText(fName[0])
        
    def openRaw(self):
        fDir = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder with raw LCMS data", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if fDir:
            self.rawPath.setText(fDir)
    
    def openResults(self):
        fName = PyQt6.QtWidgets.QFileDialog.getOpenFileName(self, "Open results table", filter="Tab separated values files (*.tsv);;Comma separated values file (*.csv);;All files (*.*)", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if fName[0]:
            self.resPath.setText(fName[0])
        
    def getUserData(self):
        return (self.expName.text(), self.traPath.text(), self.rawPath.text(), self.resPath.text(), self.delim.currentText().replace("TAB", "\t"))
    



        
                

class Window(PyQt6.QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        self._pyFilePath = os.path.join(pythonScriptDirectory)
        
        self.__sampleNameReplacements = {"Ref_": "", "METAB02_": "", "MCC025_": "", "R100140_": "", "R100138_": ""}
        self.__leftPeakDefault = -0.05
        self.__rightPeakDefault = 0.05
        self.__defaultSampleOrder = ['_CAL[0-9]+_', '_NIST[0-9]+_', '_BLK[0-9]+_', '_QC[0-9]*_', '_SST[0-9]*_', '.*']
        self.__normalColor     = (112, 128, 144)
        self.__highlightColor1 = (178,  34,  34)
        self.__highlightColor2 = (255, 165,   0)
        self.__msConvertPath = "msconvert" #"%LOCALAPPDATA%\\Apps\\ProteoWizard 3.0.22119.ba94f16 32-bit\\msconvert.exe"
        self.__calibrationFunctionstep = 100
        self.__exportSeparator = "\t"
        self.__defaultJumpWidth = 0.005
        self.__areaFormatter = "%6.4e"
        self.__icons = {"res/PB/peak": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "res_PB_peak.png")),
                        "res/PB/noise": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "res_PB_noise.png")),
                        "res/PB/nothing": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "res_PB_nothing.png")),
                        
                        "res/manual/peak": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "res_manual_peak.png")),
                        "res/manual/noise": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "res_manual_noise.png")),
                        "res/manual/nothing": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "res_manual_nothing.png")),
                        
                        "other": PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "other.png")),
                        None: PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "error.png"))
                        }
        
        if not os.path.exists(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM")):
            os.mkdir(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM"))
        if not os.path.exists(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models")):
            os.mkdir(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models"))
                              
        
        self.setWindowTitle("PeakBotMRM (version '%s')"%(PeakBotMRM.Config.VERSION))
        self.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
        
        self._plots = []
        self.dockArea = DockArea()
        self.docks = []
        for i in range(9):
            plot = self.genPlot()
            dock = Dock(["Sub EIC", "Sub EICs", "Sub peaks", "ISTD EIC", "ISTD EICs", "ISTD peaks", "Sub calibration", "ISTD calibration", "Sub / ISTD calibration"][i], autoOrientation=False)
            dock.setOrientation('vertical', force=True)
            self.docks.append(dock)
            dock.addWidget(plot)
            self.dockArea.addDock(dock)
        self.dockArea.moveDock(self.docks[1], "right", self.docks[0])
        self.dockArea.moveDock(self.docks[2], "right", self.docks[1])
        self.dockArea.moveDock(self.docks[4], "right", self.docks[3])
        self.dockArea.moveDock(self.docks[5], "right", self.docks[4])
        self.dockArea.moveDock(self.docks[7], "right", self.docks[6])
        self.dockArea.moveDock(self.docks[8], "right", self.docks[7])
        
        self.infoLabel = PyQt6.QtWidgets.QLabel("")
        grid = PyQt6.QtWidgets.QGridLayout()
        grid.setContentsMargins(0,0,0,0)
        grid.addWidget(self.infoLabel, 0, 0)
        self.hasPeak = PyQt6.QtWidgets.QComboBox()
        self.hasPeak.addItems(["PB - Nothing", "PB - Peak", "PB - Noise", "Manual - Nothing", "Manual - Peak", "Manual - Noise"])
        self.peakStart = PyQt6.QtWidgets.QDoubleSpinBox()
        self.peakStart.setDecimals(3); self.peakStart.setMaximum(100); self.peakStart.setMinimum(0); self.peakStart.setSingleStep(0.005)
        self.peakEnd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.peakEnd.setDecimals(3); self.peakEnd.setMaximum(100); self.peakEnd.setMinimum(0); self.peakEnd.setSingleStep(0.005)
        self.istdhasPeak = PyQt6.QtWidgets.QComboBox()
        self.istdhasPeak.addItems(["PB - Nothing", "PB - Peak", "PB - Noise", "Manual - Nothing", "Manual - Peak", "Manual - Noise"])
        self.istdpeakStart = PyQt6.QtWidgets.QDoubleSpinBox()
        self.istdpeakStart.setDecimals(3); self.istdpeakStart.setMaximum(100); self.istdpeakStart.setMinimum(0); self.istdpeakStart.setSingleStep(0.005)
        self.istdpeakEnd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.istdpeakEnd.setDecimals(3); self.istdpeakEnd.setMaximum(100); self.istdpeakEnd.setMinimum(0); self.istdpeakEnd.setSingleStep(0.005)
        self.useForCalibration = PyQt6.QtWidgets.QCheckBox()
        self.calibrationMethod = PyQt6.QtWidgets.QComboBox()
        self.calibrationMethod.addItems(["linear", "linear, 1/expConc.", "quadratic", "quadratic, 1/expConc."])
        self.calibrationMethod.setCurrentIndex(1)
        layout = PyQt6.QtWidgets.QHBoxLayout()
        
        layout.addWidget(PyQt6.QtWidgets.QLabel("Substance:"))
        layout.addWidget(self.hasPeak)
        self.hasPeak.currentIndexChanged.connect(self.featurePropertiesChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("start (min)"))
        layout.addWidget(self.peakStart)
        self.peakStart.valueChanged.connect(self.featurePropertiesChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("end (min)"))
        layout.addWidget(self.peakEnd)
        self.peakEnd.valueChanged.connect(self.featurePropertiesChanged)
        layout.addStretch()
        
        layout.addWidget(PyQt6.QtWidgets.QLabel("ISTD:"))
        layout.addWidget(self.istdhasPeak)
        self.istdhasPeak.currentIndexChanged.connect(self.featurePropertiesChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("start (min)"))
        layout.addWidget(self.istdpeakStart)
        self.istdpeakStart.valueChanged.connect(self.featurePropertiesChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("end (min)"))
        layout.addWidget(self.istdpeakEnd)
        self.istdpeakEnd.valueChanged.connect(self.featurePropertiesChanged)
        layout.addStretch()
        
        layout.addWidget(PyQt6.QtWidgets.QLabel("Use for calibration"))
        layout.addWidget(self.useForCalibration)
        self.useForCalibration.stateChanged.connect(self.featurePropertiesChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("Method"))
        layout.addWidget(self.calibrationMethod)
        self.calibrationMethod.currentIndexChanged.connect(self.curInterpolationFunctionChanged)
        
        grid.addLayout(layout, 1, 0)
        helper = PyQt6.QtWidgets.QWidget()
        helper.setLayout(grid)
        dock = Dock("Ins. Props", size=(1,1), hideTitle=True)
        self.docks.append(dock)
        dock.addWidget(helper)
        self.dockArea.addDock(dock, "top")        
        
        self.sortOrder = PyQt6.QtWidgets.QComboBox()
        self.sortOrder.addItems(["Sort: Sample group/name", "Sort: Peak area asc.", "Sort: Peak area desc."])
        self.sortOrder.setCurrentIndex(0)
        self.sortOrder.currentIndexChanged.connect(self.resortExperiments)
        temp = PyQt6.QtWidgets.QWidget()
        helper = PyQt6.QtWidgets.QHBoxLayout()
        helper.setContentsMargins(0,0,0,0)
        helper.addWidget(PyQt6.QtWidgets.QLabel("Filter"))
        self.treeFilter = PyQt6.QtWidgets.QLineEdit("samp:")
        self.treeFilter.textChanged.connect(self.filterUpdated)
        helper.addWidget(self.treeFilter)
        temp.setLayout(helper)
        self.tree = PyQt6.QtWidgets.QTreeWidget()
        self.tree.setSelectionMode(PyQt6.QtWidgets.QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self.treeSelectionChanged)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Generic", "Area", "PeakWidth"])
        self.tree.setMinimumWidth(300)
        self.tree.header().resizeSection(0, 250)
        self.tree.header().resizeSection(1, 50)
        self.tree.header().resizeSection(2, 50)
        dock = Dock("Experiments")
        self.docks.append(dock)
        dock.addWidget(temp)
        dock.addWidget(self.sortOrder)
        dock.addWidget(self.tree)
        dock.setStretch(y = 23)
        self.dockArea.addDock(dock, "left")
        
        plot = self.genPlot()
        dock = Dock("PaCMAP embedding - all EICs")
        self.docks.append(dock)
        dock.addWidget(plot)
        self.dockArea.addDock(dock)
        self.dockArea.moveDock(self.docks[11], "bottom", self.docks[10])
        
        self.setCentralWidget(self.dockArea)
        
        ## Toolbar
        toolbar = PyQt6.QtWidgets.QToolBar("Tools")
        toolbar.setIconSize(PyQt6.QtCore.QSize(16, 16))
        self.addToolBar(PyQt6.QtCore.Qt.ToolBarArea.LeftToolBarArea, toolbar)
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Open"))
        
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "folder-open-outline.svg")), "New / open", self)
        item.triggered.connect(self.userSelectExperimentalData)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "load-outline.svg")), "Open (binary experiment)", self)
        item.triggered.connect(self.loadBinaryExperimentHelper)
        toolbar.addAction(item)
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Edit"))

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "create-outline.svg")), "Edit experiment meta data", self)
        item.triggered.connect(self.editExperimentMetaData)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "thunderstorm-outline.svg")), "Reset instances", self)
        item.triggered.connect(self.resetActivateExperiment)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "close-circle-outline.svg")), "Close experiment", self)
        item.triggered.connect(self.closeExperiment)
        toolbar.addAction(item)
        
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Process"))

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")), "Run PeakBotMRM detection on active experiment", self)
        item.triggered.connect(self.processActiveExperimentEventHelper)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robots.png")), "Run PeakBotMRM detection on all openend experiments", self)
        item.triggered.connect(self.processAllExperimentEventHelper)
        toolbar.addAction(item)
        
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Save"))

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "save-outline.svg")), "Save (binary experiment)", self)
        item.triggered.connect(self.saveBinaryExperimentHelper)
        toolbar.addAction(item)
        
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Export"))

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "list-circle-outline.svg")), "Show summary of active experiment", self)
        item.triggered.connect(functools.partial(self.showSummary, processingInfo = None))
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "download-outline.svg")), "Export active experiment results", self)
        item.triggered.connect(functools.partial(self.exportIntegrations, all = False))
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "document-text-outline.svg")), "Export active experiment report", self)
        item.triggered.connect(functools.partial(self.exportReport))
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "code-download-outline.svg")), "Export all experiment results", self)
        item.triggered.connect(functools.partial(self.exportIntegrations, all = True))
        toolbar.addAction(item)
        
        ## TODO implement
        if False:
            toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
            toolbar.addWidget(PyQt6.QtWidgets.QLabel("Train"))
            
            item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot_grey.png")), "Train new model with active experiment", self)
            item.triggered.connect(self.trainNewModel)
            toolbar.addAction(item)
        
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Misc"))
        
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "settings-outline.svg")), "Settings", self)
        item.triggered.connect(self.showSettings)
        toolbar.addAction(item)
        
        temp = self.dockArea.saveState()
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "grid-outline.svg")), "Restore standard GUI layout", self)
        item.triggered.connect(lambda: self.dockArea.restoreState(temp))
        toolbar.addAction(item)
        
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "refresh-outline.svg")), "Refresh all views", self)
        item.triggered.connect(self.refreshViews)
        toolbar.addAction(item)
        
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "crop-outline.svg")), "Add polygon ROI to PaCMAP embedding", self)
        item.triggered.connect(self.addPolyROItoPaCMAP)
        toolbar.addAction(item)
        
        
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("    "))
        toolbar.addWidget(PyQt6.QtWidgets.QLabel("Info",))

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "information-circle-outline.svg")), "Information", self)
        item.triggered.connect(self.showPeakBotMRMInfo)
        toolbar.addAction(item)
        
        
        self.loadedExperiments = {}
        
        self.lastExp = ""
        self.lastSub = ""
        self.lastSam = ""
        
        self.paCMAP = None
        
        self.tree._keyPressEvent = self.tree.keyPressEvent
        self.tree.keyPressEvent = self.keyPressEvent
            
        if os.path.exists(os.path.join(self._pyFilePath, "defaultSettings.pickle")):
            self.tree.blockSignals(True)
            self.loadSettingsFromFile(settingsFile = os.path.join(self._pyFilePath, "defaultSettings.pickle"))
            self.tree.blockSignals(False)

        try:
            subprocess.run(self.__msConvertPath, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as ex:
            logging.error("\033[91mError: msconvert (%s) not found.\033[0m Download and install from https://proteowizard.sourceforge.io/")
            PyQt6.QtWidgets.QMessageBox.critical(None, "PeakBotMRM", "Error<br><br>MSConvert (at '%s') cannot be found. Please verify that it is present/installed and/or set the path to the executible accordingly in the settings<br><br>Download MSconvert from <a href='https://proteowizard.sourceforge.io/'>https://proteowizard.sourceforge.io/</a>.<br>Choose the version that is 'able to convert ventdor files'.<br>Install the software.<br>Then try restarting PeakBotMRM. If 'msconvert' alone does not work, try '%%LOCALAPPDATA%%\\Apps\\ProteoWizard 3.0.22119.ba94f16 32-bit\\msconvert.exe' (and/or replace the version with the one you have installed)"%(self.__msConvertPath))
    
    def filterUpdated(self):
        filter = self.treeFilter.text()
        filter = "" if filter is None else filter
        
        if ":" in filter:
            typ = filter[:filter.index(":")]
            fil = filter[(filter.index(":")+1):]
            
            if typ.lower() == "samp":
                for iti in range(self.tree.topLevelItemCount()):
                    it = self.tree.topLevelItem(iti)
                    for subi in range(it.childCount()):
                        subit = it.child(subi)
                        for sampi in range(subit.childCount()):
                            sampit = subit.child(sampi)
                            sampit.setHidden(not (fil in sampit.text(0) or fil == ""))
                                
    
    def keyPressEvent(self, event):
        if event.key() in [PyQt6.QtCore.Qt.Key.Key_Down, PyQt6.QtCore.Qt.Key.Key_Up]:
            self.tree._keyPressEvent(event)
            
        if 32 <= event.key() <= 126:
            modifiers = PyQt6.QtWidgets.QApplication.keyboardModifiers()
            
            if chr(event.key()) in ["Q"]:
                self.hasPeak.setCurrentIndex({0:0, 1:1, 2:2, 128:3, 129:4, 130:5}[((self.hasPeak.currentIndex() % 128 + 1) % 3) + 128])
            elif chr(event.key()) in ["W"]:
                self.peakStart.setValue(self.peakStart.value() - self.__defaultJumpWidth)
            elif chr(event.key()) in ["E"]:
                self.peakStart.setValue(self.peakStart.value() + self.__defaultJumpWidth)
            elif chr(event.key()) in ["R"]:
                self.peakEnd.setValue(self.peakEnd.value() - self.__defaultJumpWidth)
            elif chr(event.key()) in ["T"]:
                self.peakEnd.setValue(self.peakEnd.value() + self.__defaultJumpWidth)
            
            if modifiers == PyQt6.QtCore.Qt.KeyboardModifier.ShiftModifier:
                if chr(event.key()) in ["A"]:
                    self.istdhasPeak.setCurrentIndex({0:0, 1:1, 2:2, 128:3, 129:4, 130:5}[((self.istdhasPeak.currentIndex() % 128 + 1) % 3) + 128])
                elif chr(event.key()) in ["S"]:
                    self.istdpeakStart.setValue(self.istdpeakStart.value() - self.__defaultJumpWidth)
                elif chr(event.key()) in ["D"]:
                    self.istdpeakStart.setValue(self.istdpeakStart.value() + self.__defaultJumpWidth)
                elif chr(event.key()) in ["F"]:
                    self.istdpeakEnd.setValue(self.istdpeakEnd.value() - self.__defaultJumpWidth)
                elif chr(event.key()) in ["G"]:
                    self.istdpeakEnd.setValue(self.istdpeakEnd.value() + self.__defaultJumpWidth)
        
    def _getSaveSettingsObject(self):
        settings = {
                    "PeakBotMRM.Config.RTSLICES": PeakBotMRM.Config.RTSLICES,
                    "PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN": PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN,
                    "PeakBotMRM.Config.INTEGRATIONMETHOD": PeakBotMRM.Config.INTEGRATIONMETHOD,
                    "PeakBotMRM.Config.CALIBRATIONMETHOD": PeakBotMRM.Config.CALIBRATIONMETHOD,
                    "PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT": PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT, 
                    "PeakBotMRM.Config.MRMHEADER": PeakBotMRM.Config.MRMHEADER,
                    "PeakBotMRM.Config.INTEGRATENOISE": PeakBotMRM.Config.INTEGRATENOISE,
                    "PeakBotMRM.Config.INTEGRATENOISE_StartQuantile": PeakBotMRM.Config.INTEGRATENOISE_StartQuantile,
                    "PeakBotMRM.Config.INTEGRATENOISE_EndQuantile": PeakBotMRM.Config.INTEGRATENOISE_EndQuantile,
                    
                    "GUI/__sampleNameReplacements": self.__sampleNameReplacements,
                    "GUI/__leftPeakDefault": self.__leftPeakDefault,
                    "GUI/__rightPeakDefault": self.__rightPeakDefault,
                    "GUI/__defaultSampleConfig": self.__defaultSampleOrder,
                    "GUI/__normalColor": self.__normalColor,
                    "GUI/__highlightColor1": self.__highlightColor1,
                    "GUI/__highlightColor2": self.__highlightColor2,
                    "GUI/__calibrationFunctionstep": self.__calibrationFunctionstep, 
                    "GUI/__exportSeparator": self.__exportSeparator.replace("\t", "TAB"), 
                    "GUI/__sortOrder": self.sortOrder.itemText(self.sortOrder.currentIndex()),
                    "GUI/__defaultJumpWidth": self.__defaultJumpWidth, 
                    "GUI/__areaFormatter": self.__areaFormatter, 
                    
                    "GUI/DockAreaState": self.dockArea.saveState(),
                }
        return settings
    
    def _loadSettingsFromObject(self, settings):
        PeakBotMRM.Config.RTSLICES = settings["PeakBotMRM.Config.RTSLICES"]
        PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN = settings["PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN"]
        PeakBotMRM.Config.INTEGRATIONMETHOD = settings["PeakBotMRM.Config.INTEGRATIONMETHOD"]
        PeakBotMRM.Config.CALIBRATIONMETHOD = settings["PeakBotMRM.Config.CALIBRATIONMETHOD"]
        PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT = settings["PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT"]
        PeakBotMRM.Config.MRMHEADER = settings["PeakBotMRM.Config.MRMHEADER"]
        PeakBotMRM.Config.INTEGRATENOISE = settings["PeakBotMRM.Config.INTEGRATENOISE"]
        PeakBotMRM.Config.INTEGRATENOISE_StartQuantile = settings["PeakBotMRM.Config.INTEGRATENOISE_StartQuantile"]
        PeakBotMRM.Config.INTEGRATENOISE_EndQuantile = settings["PeakBotMRM.Config.INTEGRATENOISE_EndQuantile"]
        
        self.__sampleNameReplacements = settings["GUI/__sampleNameReplacements"]
        self.__leftPeakDefault = settings["GUI/__leftPeakDefault"]
        self.__rightPeakDefault = settings["GUI/__rightPeakDefault"]
        self.__defaultSampleOrder = settings["GUI/__defaultSampleConfig"]
        self.__normalColor = settings["GUI/__normalColor"]
        self.__highlightColor1 = settings["GUI/__highlightColor1"]
        self.__highlightColor2 = settings["GUI/__highlightColor2"]
        self.__calibrationFunctionstep = settings["GUI/__calibrationFunctionstep"]
        self.__exportSeparator = settings["GUI/__exportSeparator"].replace("TAB", "\t")
        self.sortOrder.setCurrentIndex([i for i in range(self.sortOrder.count()) if self.sortOrder.itemText(i) == settings["GUI/__sortOrder"]][0])
        self.__defaultJumpWidth = settings["GUI/__defaultJumpWidth"]
        self.peakStart.setSingleStep(self.__defaultJumpWidth)
        self.peakEnd.setSingleStep(self.__defaultJumpWidth)
        self.istdpeakStart.setSingleStep(self.__defaultJumpWidth)
        self.istdpeakEnd.setSingleStep(self.__defaultJumpWidth)
        self.__areaFormatter = settings["GUI/__areaFormatter"]
        
        #self.dockArea.restoreState(settings["GUI/DockAreaState"])

    def saveSettingsToFile(self, settingsFile = None):
        if settingsFile is None:
            settingsFile = os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "defaultSettings.pickle")
            
        with open(settingsFile, "wb") as fout:
            settings = self._getSaveSettingsObject()
            pickle.dump(settings, fout)

    def loadSettingsFromFile(self, settingsFile = None):
        if settingsFile is None:
            settingsFile = os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "defaultSettings.pickle")
            
        with open(settingsFile, "rb") as fin:
            settings = pickle.load(fin)            
            self._loadSettingsFromObject(settings)
    
    def showSettings(self):
        dialog = PyQt6.QtWidgets.QDialog(self)
        dialog.setModal(True)
        dialog.setWindowTitle("Settings")
        dialog.setFixedSize(PyQt6.QtCore.QSize(800, 700))
        
        params = [
            {'name': 'Environment', 'type': 'group', 'children': [
                {'name': 'Python', 'type': 'str', 'value': sys.version, 'readonly': True},
                {'name': 'CPU', 'type': 'str', 'value': PeakBotMRM.getCPUInfo(), 'readonly': True},
                {'name': 'Memory', 'type': 'str', 'value': PeakBotMRM.getMemoryInfo(), 'readonly': True},
                {'name': 'GPU/CUDA', 'type': 'str', 'value': PeakBotMRM.getCUDAInfo(), 'readonly': True},
                {'name': 'Tensorflow', 'type': 'str', 'value': PeakBotMRM.getTensorflowVersion(), 'readonly': True},
        ]},
            {'name': 'PeakBotMRM Configuration', 'type': 'group', 'children': [
                {'name': 'Name', 'type': 'str', 'value': PeakBotMRM.Config.NAME, 'readonly': True},
                {'name': 'version', 'type': 'str', 'value': PeakBotMRM.Config.VERSION, 'readonly': True},
                {'name': 'Standardized eic width', 'type': 'int', 'value': PeakBotMRM.Config.RTSLICES, 'limits': [1, 1001], 'step': 1},
                {'name': 'Number of classes', 'type': 'str', 'value': PeakBotMRM.Config.NUMCLASSES, 'readonly': True},
                {'name': 'Update peak borders to min. value', 'type': 'bool', 'value': PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN},
                {'name': 'Integration method', 'type': 'list', 'value': PeakBotMRM.Config.INTEGRATIONMETHOD, 'values': ['linear', 'linearbetweenborders', 'all', 'minbetweenborders']},
                {'name': 'Calibration method', 'type': 'list', 'value': PeakBotMRM.Config.CALIBRATIONMETHOD, 'values': ["linear", "linear, 1/expConc.", "quadratic", "quadratic, 1/expConc."]},
                {'name': 'Calibration extend borders', 'type': 'bool', 'value': PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT},
                {'name': 'Calibration plot step size', 'type': 'int', 'value': self.__calibrationFunctionstep, 'step': 1, 'limits': [10, 1000]},
                {'name': 'MRM header', 'type': 'str', 'value': PeakBotMRM.Config.MRMHEADER},
                {'name': 'Integrate noise', 'type': 'bool', 'value': PeakBotMRM.Config.INTEGRATENOISE},
                {'name': 'Noise integration start quantile', 'type': 'float', 'value': PeakBotMRM.Config.INTEGRATENOISE_StartQuantile, 'limits': [0, 1]},
                {'name': 'Noise integration end quantile', 'type': 'float', 'value': PeakBotMRM.Config.INTEGRATENOISE_EndQuantile, 'limits': [0, 1]},
            ]},
            {'name': 'Sample names', 'type': 'group', 'children':[
                {'name': 'Replacements', 'type': 'str', 'value': str(self.__sampleNameReplacements)},
                {'name': 'Default sample order', 'type': 'str', 'value': str(self.__defaultSampleOrder)},
            ]},
            {'name': 'New chromatographic peak (relative to ref. RT)', 'type': 'group', 'children': [
                {'name': 'Default left width', 'type': 'float', 'value': self.__leftPeakDefault, 'limits': [-1., 0], 'step' : .005, 'suffix': 'min'},
                {'name': 'Default right width', 'type': 'float', 'value': self.__rightPeakDefault, 'limits': [0., 1.], 'step' : .005, 'suffix': 'min'},
            ]},
            {'name': 'Plot colors', 'type': 'group', 'children':[
                {'name': 'Normal color', 'type': 'color', 'value': PyQt6.QtGui.QColor.fromRgb(*self.__normalColor)},
                {'name': 'Highlight color 1', 'type': 'color', 'value': PyQt6.QtGui.QColor.fromRgb(*self.__highlightColor1)},
                {'name': 'Highlight color 2', 'type': 'color', 'value': PyQt6.QtGui.QColor.fromRgb(*self.__highlightColor2)}
            ]},
            {'name': 'Other', 'type': 'group', 'children':[
                {'name': 'MSConvert executable', 'type': 'str', 'value': self.__msConvertPath, 'tip': 'Download MSconvert from <a href="https://proteowizard.sourceforge.io/">https://proteowizard.sourceforge.io/</a>. Choose the version that is "able to convert ventdor files". Install the software. Then try restarting PeakBotMRM. If "msconvert" alone does not work, try "%LOCALAPPDATA%\\Apps\\ProteoWizard 3.0.22119.ba94f16 32-bit\\msconvert.exe"'},
                {'name': 'Export delimiter', 'type': 'list', 'value': self.__exportSeparator, 'values': ["TAB", ",", ";", "$"]},
                {'name': 'Sort order', 'type': 'list', 'value': self.sortOrder.itemText(self.sortOrder.currentIndex()), 'values': [self.sortOrder.itemText(i) for i in range(self.sortOrder.count())]},
                {'name': 'Default jump width', 'type': 'float', 'value': self.__defaultJumpWidth, 'limits': [0.001, 0.2], 'suffix': 'min'},
                {'name': 'Area formatter', 'type': 'str', 'value': self.__areaFormatter, 'tip': 'Use Python string format options. Available at: <a href="https://docs.python.org/2/library/stdtypes.html#string-formatting-operations">https://docs.python.org/2/library/stdtypes.html#string-formatting-operations</a>'}
            ]},
            #{'name': 'Save/restore gui layout', 'type': 'group', 'children': [
            #    {'name': 'Save to file', 'type': 'action'},
            #    {'name': 'Load from file', 'type': 'action'},
            #]},
        ]

        ## Create tree of Parameter objects
        p = Parameter.create(name='params', type='group', children=params)
        #p.param('Save/restore gui layout', 'Save to file').sigActivated.connect(self.saveUILayout)
        #p.param('Save/restore gui layout', 'Load from file').sigActivated.connect(self.loadUILayout)
        def getSettingsFromTree():
            PeakBotMRM.Config.RTSLICES = p.param("PeakBotMRM Configuration", "Standardized eic width").value()
            PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN = p.param("PeakBotMRM Configuration", "Update peak borders to min. value").value()
            PeakBotMRM.Config.INTEGRATIONMETHOD = p.param("PeakBotMRM Configuration", "Integration method").value()
            PeakBotMRM.Config.CALIBRATIONMETHOD = p.param("PeakBotMRM Configuration", "Calibration method").value()
            PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT = p.param("PeakBotMRM Configuration", "Calibration extend borders").value()
            PeakBotMRM.Config.MRMHEADER = p.param("PeakBotMRM Configuration", "MRM header").value()
            PeakBotMRM.Config.INTEGRATENOISE = p.param("PeakBotMRM Configuration", "Integrate noise").value()
            PeakBotMRM.Config.INTEGRATENOISE_StartQuantile = p.param("PeakBotMRM Configuration", "Noise integration start quantile").value()
            PeakBotMRM.Config.INTEGRATENOISE_EndQuantile = p.param("PeakBotMRM Configuration", "Noise integration end quantile").value()
            
            if p.param("Sample names", "Replacements") is None or p.param("Sample names", "Replacements") == "":
                self.__sampleNameReplacements = {}
            else:
                try:
                    self.__sampleNameReplacements = eval(p.param("Sample names", "Replacements").value())
                    if type(self.__sampleNameReplacements) is not dict:
                        raise RuntimeError("Incorrect object specified")
                except:
                    PyQt6.QtWidgets.QMessageBox.critical(None, "PeakBotMRM", "Error<br><br>The entered sample replacements are not a valid python dictionary (e.g. {'longSampleStringThatIsNotRelevant':''}). Please modify.")
                    self.__sampleNameReplacements = {}
            if p.param("Sample names", "Default sample order") is None or p.param("Sample names", "Default sample order") == "":
                self.__defaultSampleOrder = {}
            else:
                try:
                    self.__defaultSampleOrder = eval(p.param("Sample names", "Default sample order").value())
                except:
                    PyQt6.QtWidgets.QMessageBox.critical(None, "PeakBotMRM", "Error<br><br>The entered sample importances are not a valid python dictionary (e.g. {'_CAL[0-9]+_':'Cal'}). Please modify.")
                    self.__defaultSampleOrder = ["Error, invalid python object"]
            self.__leftPeakDefault = p.param("New chromatographic peak (relative to ref. RT)", "Default left width").value()
            self.__rightPeakDefault = p.param("New chromatographic peak (relative to ref. RT)", "Default right width").value()
            self.__normalColor = p.param("Plot colors", "Normal color").value().getRgb()
            self.__highlightColor1 = p.param("Plot colors", "Highlight color 1").value().getRgb()
            self.__highlightColor2 = p.param("Plot colors", "Highlight color 2").value().getRgb()
            self.__calibrationFunctionstep = p.param("PeakBotMRM Configuration", "Calibration plot step size").value()
            self.__msConvertPath = p.param("Other", "MSConvert executable").value()
            self.__exportSeparator = p.param("Other", "Export delimiter").value().replace("TAB", "\t")
            self.sortOrder.setCurrentIndex([i for i in range(self.sortOrder.count()) if self.sortOrder.itemText(i) == p.param("Other", "Sort order").value()][0])
            self.__defaultJumpWidth = p.param("Other", "Default jump width").value()
            self.__areaFormatter = p.param("Other", "Area formatter").value()
                        
        t = ParameterTree()
        t.setParameters(p, showTop=False)
        
        layout = PyQt6.QtWidgets.QGridLayout()
        layout.addWidget(t, 0, 0, 1, 6)
        
        accept = PyQt6.QtWidgets.QPushButton("Accept")
        accept.clicked.connect(dialog.accept)
        layout.addWidget(accept, 1, 5, 1, 1)
        
        reject = PyQt6.QtWidgets.QPushButton("Discard")
        reject.clicked.connect(dialog.reject)
        layout.addWidget(reject, 1, 4, 1, 1)
        
        loadDefault = PyQt6.QtWidgets.QPushButton("Load default")
        def action():
            dialog.reject()
            self.loadSettingsFromFile()
            self.showSettings()
        loadDefault.clicked.connect(action)
        layout.addWidget(loadDefault, 1, 0, 1, 1)
        
        saveAsDefault = PyQt6.QtWidgets.QPushButton("Accept and save as default")
        def action():
            dialog.accept()
            self.saveSettingsToFile()
        saveAsDefault.clicked.connect(action)
        layout.addWidget(saveAsDefault, 1, 1, 1, 2)
        
        dialog.setLayout(layout)
        
        ## TODO load and save layout
        ## TODO transfer settings from and to window
        
        if dialog.exec():
            getSettingsFromTree()
    
    def saveUILayout(self):
        print("save gui layout")
    
    def loadUILayout(self):
        print("load ui layout")
    
    def showPeakBotMRMInfo(self):
        PyQt6.QtWidgets.QMessageBox.information(self, "PeakBotMRM", "<b>PeakBotMRM</b><br><br>PeakBotMRM was developed at <a href='https://cemm.at/research/facilities/molecular-discovery-platform/metabolomics-facility'>CEMM</a> and at the <a href='https://chemnet.univie.ac.at/'>University of Vienna</a>.<br> For further information please contact the authors.<br>(c) 2020 - 2022<br><br><b>Commercial use is prohibited!</b><br><br>Figures and illustrations have been desigend using resources from <a href='https://flaticon.com'>https://flaticon.com</a> and <a href='https://ionic.io/ionicons'>https://ionic.io/ionicons</a>")
    
    def processAllExperimentEventHelper(self):
        self.processExperimentSHelper(all = True)        
    
    def processActiveExperimentEventHelper(self):
        self.processExperimentSHelper(all = False)
        
    def trainNewModel(self):
        x = TrainModelDialog(self)
        x.setModal(True)
        x.exec()
        PyQt6.QtWidgets.QMessageBox.information(self, "PeakBotMRM", "<b>Train a new model</b><br><br>This function is not yet implemented...")
        
    def processExperimentSHelper(self, all = False):
        menu = PyQt6.QtWidgets.QMenu(self)
        
        addSep = False
        for mf in natsort.natsorted(list(os.listdir(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models"))), key = lambda x: x.lower()):
            if mf.endswith(".h5"):
                acc = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")), mf, self)
                acc.triggered.connect(functools.partial(self.processExperimentS, os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models", mf), all = all))
                menu.addAction(acc)
                addSep = True
        
        if addSep: 
            menu.addSeparator()
        acc = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "folder-open-outline.svg")), "Open other model", self)
        acc.triggered.connect(functools.partial(self.processExperimentS, None, all = all))
        menu.addAction(acc)
            
        menu.exec(PyQt6.QtGui.QCursor.pos())
    
    def processExperimentS(self, peakBotMRMModelFile = None, all = False, keepManualIntegrations = None):
                
        if peakBotMRMModelFile is None:
            peakBotMRMModelFile = PyQt6.QtWidgets.QFileDialog.getOpenFileName(self, "Open PeakBotMRM model file", filter="PeakBotMRM models (*.h5)", directory=os.path.join(self._pyFilePath, "models"), options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
            if peakBotMRMModelFile[0]:
                peakBotMRMModelFile = peakBotMRMModelFile[0]
                button = PyQt6.QtWidgets.QMessageBox.question(self, "New model", "Do you want this model to be copied to your default models for easy and quick access?")
                if button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    shutil.copyfile(peakBotMRMModelFile, os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models", os.path.basename(peakBotMRMModelFile)))
                    peakBotMRMModelFile = os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models", os.path.basename(peakBotMRMModelFile))                
            else:
                return
            
        expToProcess = []
        if all:
            for tlItemInd in range(self.tree.topLevelItemCount()):
                treeItem = self.tree.topLevelItem(tlItemInd)
                selExp = treeItem.experiment if "experiment" in treeItem.__dict__ else None
            
                if selExp is not None and selExp != "" and selExp in self.loadedExperiments:
                    expToProcess.append(selExp)
                
        else:
            l = list(self.tree.selectedItems())
            
            if len(l) == 1 and "experiment" in l[0].__dict__:
                treeItem = l[0]
                while treeItem.parent() is not None:
                    treeItem = treeItem.parent()
            
                selExp = treeItem.experiment if "experiment" in treeItem.__dict__ else None
            
                if selExp is not None and selExp != "" and selExp in self.loadedExperiments:
                    expToProcess.append(selExp)
        
        button = None
        if keepManualIntegrations is None:
            button = PyQt6.QtWidgets.QMessageBox.question(self, "Process experiment", "<b>Warning</b><br><br>This will process the selected experiment(s) <br>'%s'<br> with PeakBotMRM (model '%s').<br> Do you want to keep any manual integrations?"%("', '".join(expToProcess), peakBotMRMModelFile), buttons = PyQt6.QtWidgets.QMessageBox.StandardButton.Yes | PyQt6.QtWidgets.QMessageBox.StandardButton.No | PyQt6.QtWidgets.QMessageBox.StandardButton.Abort, defaultButton = PyQt6.QtWidgets.QMessageBox.StandardButton.Yes)
        else:
            button = PyQt6.QtWidgets.QMessageBox.StandardButton.Yes if keepManualIntegrations else PyQt6.QtWidgets.QMessageBox.StandardButton.No
            
        if button in [PyQt6.QtWidgets.QMessageBox.StandardButton.Yes, PyQt6.QtWidgets.QMessageBox.StandardButton.No]:
            
            for selExp in expToProcess:
                
                prDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Predicting experiment '%s' with PeakBotMRM<br>Model: '%s'<br><br>See console for further details"%(selExp, os.path.basename(peakBotMRMModelFile)), 
                                                            minimum=0, maximum=len(self.loadedExperiments[selExp].integrations))
                prDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
                prDiag.setWindowTitle("PeakBotMRM")
                prDiag.setModal(True)
                prDiag.show()
                logging.info("")
                logging.info("")
                logging.info("Processing dataset '%s'"%(selExp))
                PeakBotMRM.predict.predictDataset(peakBotMRMModelFile, self.loadedExperiments[selExp].substances, self.loadedExperiments[selExp].integrations, callBackFunction = prDiag.setValue, showConsoleProgress = False)
                
                prDiag.hide()
                
                for sub in self.loadedExperiments[selExp].integrations:
                    for samp in self.loadedExperiments[selExp].integrations[sub]:
                        if (button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes and self.loadedExperiments[selExp].integrations[sub][samp].type != "Manual integration") or button == PyQt6.QtWidgets.QMessageBox.StandardButton.No:
                            self.loadedExperiments[selExp].integrations[sub][samp].type = self.loadedExperiments[selExp].integrations[sub][samp].other["pred.type"]
                            self.loadedExperiments[selExp].integrations[sub][samp].comment = self.loadedExperiments[selExp].integrations[sub][samp].other["pred.comment"]
                            
                            self.loadedExperiments[selExp].integrations[sub][samp].foundPeak = self.loadedExperiments[selExp].integrations[sub][samp].other["pred.foundPeak"]
                            self.loadedExperiments[selExp].integrations[sub][samp].rtStart = self.loadedExperiments[selExp].integrations[sub][samp].other["pred.rtstart"]
                            self.loadedExperiments[selExp].integrations[sub][samp].rtEnd = self.loadedExperiments[selExp].integrations[sub][samp].other["pred.rtend"]
                            self.loadedExperiments[selExp].integrations[sub][samp].area = self.loadedExperiments[selExp].integrations[sub][samp].other["pred.areaPB"]
            
                for tlItemInd in range(self.tree.topLevelItemCount()):
                    treeItem = self.tree.topLevelItem(tlItemInd)
                    temp = treeItem.experiment if "experiment" in treeItem.__dict__ else None
                    if temp == selExp:
                        treeItem.setBackground(0, PyQt6.QtGui.QColor.fromRgb(255,255,255))
                        for subNodeInd in range(treeItem.childCount()):
                            subNode = treeItem.child(subNodeInd)
                            for sampNodeInd in range(subNode.childCount()):
                                sampleItem = subNode.child(sampNodeInd)
                                inte = self.loadedExperiments[selExp].integrations[sampleItem.substance][sampleItem.sample]
                                sampleItem.setIcon(0, {0: self.__icons["res/PB/nothing"], 1: self.__icons["res/PB/peak"], 2: self.__icons["res/PB/noise"], 128: self.__icons["res/manual/nothing"], 129: self.__icons["res/manual/peak"], 130: self.__icons["res/manual/noise"]}[inte.foundPeak])
                                sampleItem.setText(1, self.__areaFormatter%(inte.area) if inte.foundPeak else "")
                                sampleItem.setText(2, "%.2f - %.2f"%(inte.rtStart, inte.rtEnd) if inte.foundPeak else "")
            
            if len(expToProcess) == 1:
                if PyQt6.QtWidgets.QMessageBox.question(self, "Generate summary", "Do you want to generate a summary of the processed results and detected chromatographic peaks?") == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    processingInfo = ["PeakBotMRM model file: %s"%(peakBotMRMModelFile)]
                    processingInfo.extend(PeakBotMRM.Config.getAsStringFancy().split("\n"))
                    self.showSummary(processingInfo = processingInfo)
            else:    
                l = list(self.tree.selectedItems())
                if len(l) == 1 and "experiment" in l[0].__dict__:
                    PyQt6.QtWidgets.QMessageBox.information(self, "Processed experiment(s)", "Experiment(s) has/have been processed. ")
            
    def editExperimentMetaData(self):
        
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "experiment" in l[0].__dict__:
            it = l[0]
            while it.parent() is not None:
                it = it.parent()
            
            selExp = it.experiment if "experiment" in it.__dict__ else None
            
            if selExp is not None:
                data = []
                for samp in self.loadedExperiments[selExp].sampleInfo:
                    temp = {}
                    temp["File name"] = samp
                    for k in self.loadedExperiments[selExp].sampleInfo[samp]:
                        if not k in ["File name", "path", "converted", "Acq. Date-Time", "Method", "Name", "Data File"]:
                            temp[k] = self.loadedExperiments[selExp].sampleInfo[samp][k]
                    data.append(temp)
                    
            x = EditTableDialog(self, data = data)
            x.setModal(True)
            x.exec()
            
            headers, dat = x.getUserData()
            for samp in self.loadedExperiments[selExp].sampleInfo:
                temp = [x for x in dat if x["File name"] == samp]
                assert len(temp) == 1
                temp = temp[0]
                
                for k in headers:
                    if not k in ["File name", "path", "converted", "Acq. Date-Time", "Method", "Name", "Data File"]:
                        if not k in self.loadedExperiments[selExp].sampleInfo[samp]:
                            self.loadedExperiments[selExp].sampleInfo[samp][k] = ""
                        if k in temp and self.loadedExperiments[selExp].sampleInfo[samp][k] != temp[k]:
                            
                            self.loadedExperiments[selExp].sampleInfo[samp][k] = temp[k]
            
    
    def showSummary(self, processingInfo = None):
        top = """
<!DOCTYPE html>
<html>
  <head>
    <title>Overview of results</title>
    <style>
      /* CSS styles */
      * {
        margin-left: 40px;
        font-family: Arial;
      }      
      pre {
          font-family: Consolas,monospace
      }      
      h1, h2, h3 {
        font-family: Arial;
        color: Slategrey;
        border: solid 1px   #bdc3c7; 
        border-radius: 3px;
        padding: 10px;
        padding-left: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.26);
        margin-top: 100px;
        margin-left: -30px;
        margin-right: 30px;
        transform: skew(-21deg);
      }      
      h1 > p, h2 > p, h3 > p {
        transform: skew(21deg);
        margin: 0px;
        margin-left: 10px;
      }      
      h1 {
        border-left: solid 6px #a93226; 
      }      
      h2 {
        border-left: solid 6px #f39c12; 
      }      
      h3 {
        border-left: solid 6px #2980b9; 
      }
    </style>
  </head>
  <body>"""
        bot = """
  </body>
</html>"""
        body = []
        
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "experiment" in l[0].__dict__:
            with tempfile.TemporaryDirectory() as tmpDir:
                it = l[0]
                while it.parent() is not None:
                    it = it.parent()
                
                selExp = it.experiment if "experiment" in it.__dict__ else None
                
                procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Generating summary of '%s'"%(selExp))
                procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
                procDiag.setWindowTitle("PeakBotMRM")
                procDiag.setModal(True)
                procDiag.show()

                body.append("<h1><p>Results of %s</p></h1>"%(selExp))
                
                if processingInfo is not None:
                    body.append("<h2><p>Processing info</p></h2>")
                    body.append("<p><ul><li>%s</li></ul></p>"%("</li><br><li>".join(processingInfo).replace("  | ..", "")))
                    
                body.append("<h2><p>Sample statistics for user-selected samples</p></h2>")
                
                ints = self.loadedExperiments[selExp].integrations
                
                dat = {"peak": [], "peakWidth": [], "rtDeviation": [], "area": [], "substance": [], "sample": []}
                colors = {}
                groups = {}
                for sub in ints:
                    refRT = self.loadedExperiments[selExp].substances[sub].refRT
                    for samp in ints[sub]:
                        if str(self.loadedExperiments[selExp].sampleInfo[samp]["use4Stats"]).lower() in ["t", "true", "1"]:
                            dat["substance"].append(sub)
                            dat["sample"].append(samp)
                            groups[samp] = self.loadedExperiments[selExp].sampleInfo[samp]["Group"]
                            colors[self.loadedExperiments[selExp].sampleInfo[samp]["Group"]] = self.loadedExperiments[selExp].sampleInfo[samp]["Color"]
                            inte = ints[sub][samp]
                            dat["peak"].append(inte.foundPeak)
                            if inte.foundPeak % 128 == 1:
                                dat["peakWidth"].append(inte.rtEnd - inte.rtStart)
                                eic = np.copy(inte.chromatogram["eic"])
                                rts = inte.chromatogram["rts"]
                                eic[rts < inte.rtStart] = 0
                                eic[rts > inte.rtEnd] = 0
                                apex = rts[np.argmax(eic)]
                                dat["rtDeviation"].append(apex - refRT)
                                dat["area"].append(inte.area)
                            else:
                                dat["peakWidth"].append(-1)
                                dat["rtDeviation"].append(0)
                                dat["area"].append(0)
                dat = pd.DataFrame(dat)
                
                temp = dat.pivot(index="sample", columns="substance", values="area").fillna(0)
                if temp.shape[0] >= 2:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    temp_norm = StandardScaler().fit_transform(temp.to_numpy())
                    temp = pd.DataFrame(temp_norm, columns = temp.columns, index = temp.index)
                    pca = PCA(n_components=2)
                    pca.fit(temp)
                    pcaScores = pca.fit_transform(temp)
                    temp = pd.DataFrame({"X": pcaScores[:,0],
                                        "Y": pcaScores[:,1],
                                        "sample": temp.index.values,
                                        "group": [groups[samp] for samp in temp.index.values]})

                    p = (p9.ggplot(data = temp, mapping = p9.aes(x="X", y="Y", label="sample", colour="group")) + p9.theme_minimal()
                        #+ p9.geom_point()
                        + p9.geom_text()
                        + p9.scales.scale_colour_manual(values = colors)
                        + p9.ggtitle("PCA") + p9.xlab("Principal component 1") + p9.ylab("Principal component 2"))
                    p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 24, height = 24, units = "in", limitsize = False)
                    body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                    body.append("<p>Note: Peak areas are used, no calibration has been carried out prior to the PCA. Z-scaling is used. Only samples with the flag 'use4Stats' are used. </p>")
                else:
                    body.append("<p>No PCA could be calculated. Please select the samples you want to include in the meta-data dialog (set the column 'use4Stats' for the samples to 'True')</p>")
                    
                body.append("<h2><p>Sample statistics for all samples</p></h2>")
                
                ints = self.loadedExperiments[selExp].integrations
                
                dat = {"peak": [], "peakWidth": [], "rtDeviation": [], "area": [], "substance": [], "sample": []}
                colors = {}
                groups = {}
                for sub in ints:
                    refRT = self.loadedExperiments[selExp].substances[sub].refRT
                    for samp in ints[sub]:
                        dat["substance"].append(sub)
                        dat["sample"].append(samp)
                        groups[samp] = self.loadedExperiments[selExp].sampleInfo[samp]["Group"]
                        colors[self.loadedExperiments[selExp].sampleInfo[samp]["Group"]] = self.loadedExperiments[selExp].sampleInfo[samp]["Color"]
                        inte = ints[sub][samp]
                        dat["peak"].append({0: "Nothing", 1: "Peak", 2: "Noise integration", 128: "Nothing - manual", 129: "Peak - manual", 130: "Noise integration - manual"}[inte.foundPeak])
                        if inte.foundPeak % 128:
                            dat["peakWidth"].append(inte.rtEnd - inte.rtStart)
                            eic = np.copy(inte.chromatogram["eic"])
                            rts = inte.chromatogram["rts"]
                            eic[rts < inte.rtStart] = 0
                            eic[rts > inte.rtEnd] = 0
                            apex = rts[np.argmax(eic)]
                            dat["rtDeviation"].append(apex - refRT)
                            dat["area"].append(inte.area)
                        else:
                            dat["peakWidth"].append(-1)
                            dat["rtDeviation"].append(0)
                            dat["area"].append(0)
                dat = pd.DataFrame(dat)
                dat['peak'] = dat['peak'].astype(object)
                
                temp = dat.pivot(index="sample", columns="substance", values="area").fillna(0)

                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                temp_norm = StandardScaler().fit_transform(temp.to_numpy())
                temp = pd.DataFrame(temp_norm, columns = temp.columns, index = temp.index)
                pca = PCA(n_components=2)
                pca.fit(temp)
                pcaScores = pca.fit_transform(temp)
                temp = pd.DataFrame({"X": pcaScores[:,0],
                                     "Y": pcaScores[:,1],
                                     "sample": temp.index.values,
                                     "group": [groups[samp] for samp in temp.index.values]})

                p = (p9.ggplot(data = temp, mapping = p9.aes(x="X", y="Y", label="sample", colour="group")) + p9.theme_minimal()
                    #+ p9.geom_point()
                    + p9.geom_text()
                    + p9.scales.scale_colour_manual(values = colors)
                    + p9.ggtitle("PCA") + p9.xlab("Principal component 1") + p9.ylab("Principal component 2"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 24, height = 24, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                body.append("<p>Note: Peak areas are used, no calibration has been carried out prior to the PCA. Z-scaling is used. Blank, calibration, STD samples are also included. </p>")
                
                                
                body.append("<h2><p>Peak statistics</p></h2>")
                df = pd.DataFrame({"type": ["Other", "Peak", "Noise", "Manual"], "value": [np.sum(dat["peak"] == "Other"), np.sum(dat["peak"] == "Peak"), np.sum(dat["peak"] == "Noise"), np.sum(dat["peak"] == "Manual")]})
                p = (p9.ggplot(data = df, mapping = p9.aes(x="type", y="value")) + p9.theme_minimal()
                     + p9.geom_bar(stat="identity") 
                     + p9.coord_flip()
                     + p9.ggtitle("Detected peaks")
                     + p9.xlab("Number of instances")
                     + p9.ylab(""))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                ## Peak width
                body.append("<h3><p>Peak width</p></h3>")
                p = (p9.ggplot(data = dat[dat["peak"] == "Peak"], mapping = p9.aes(x="peakWidth", fill = "peak")) + p9.theme_minimal()
                    + p9.geom_histogram(bins = 100, position = 'identity')
                    + p9.ggtitle("Peak width (min)")
                    + p9.xlab("Peak width (min)")
                    + p9.ylab("Number of instances"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
            
                p = (p9.ggplot(data = dat[dat["peak"] == "Peak"], mapping = p9.aes(x="peakWidth", fill = "peak")) + p9.theme_minimal()
                    + p9.geom_histogram(bins = 100, position = 'identity')
                    + p9.xlim([0, 2])
                    + p9.ggtitle("Peak width zoomed (min)")
                    + p9.xlab("Peak width (min)")
                    + p9.ylab("Number of instances"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                orde = list(dat[dat["peak"] == "Peak"].groupby(["peak", "substance"]).mean().sort_values(["peakWidth"], ascending = False).reset_index()["substance"])
                for sub in dat["substance"]:
                    if sub not in orde:
                        orde.append(sub)
                p = (p9.ggplot(data = dat[dat["peak"] != "Other"], mapping = p9.aes(x="peakWidth", y="substance", colour = "peak")) + p9.theme_minimal()
                    + p9.geom_jitter(alpha = 0.1, width = 0, height = 0.2)
                    + p9.xlim([0, 2])
                    + p9.scales.scale_y_discrete(limits=orde)
                    + p9.ggtitle("Peak width zoomed (min)")
                    + p9.xlab("Peak width (min)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 32/250*len(ints), units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                p = (p9.ggplot(data = dat[dat["peak"] != "Other"], mapping = p9.aes(x="peakWidth", y="area", colour = "peak")) + p9.theme_minimal()
                    + p9.geom_point(alpha = 0.1)
                    + p9.xlim([0, 2])
                    + p9.scales.scale_y_log10()
                    + p9.ggtitle("Peak width vs. area, zoomed (min)")
                    + p9.xlab("Peak width (min)")
                    + p9.ylab("Peak area (log10)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 18, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                                
                
                ## Peak area
                body.append("<h3><p>Peak areas</p></h3>")
                p = (p9.ggplot(data = dat[dat["peak"] != "Other"], mapping = p9.aes(x="area", fill = "peak")) + p9.theme_minimal()
                    + p9.geom_histogram(bins = 1000, position = 'identity')
                    + p9.scales.scale_x_log10()
                    + p9.ggtitle("Peak areas")
                    + p9.xlab("Peak area")
                    + p9.ylab("Number of instances"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                orde = list(dat[dat["peak"] == "Peak"].groupby(["peak", "substance"]).mean().sort_values(["area"], ascending = False).reset_index()["substance"])
                for sub in dat["substance"]:
                    if sub not in orde:
                        orde.append(sub)
                p = (p9.ggplot(data = dat[dat["peak"] != "Other"], mapping = p9.aes(x="area", y = "substance", colour = "peak")) + p9.theme_minimal()
                    + p9.geom_point(alpha=0.2)
                    + p9.scales.scale_y_discrete(limits=orde)
                    + p9.scales.scale_x_log10()
                    + p9.ggtitle("Peak areas"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 32/250*len(ints), units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                
                ## Peak apex relative to reference rt
                body.append("<h3><p>Peak retention times</p></h3>")
                p = (p9.ggplot(data = dat[dat["peak"] == "Peak"], mapping = p9.aes(x="rtDeviation")) + p9.theme_minimal()
                    + p9.geom_histogram(bins = 100)
                    + p9.ggtitle("Peak deviation (min; apex - reference)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                orde = list(dat[dat["peak"] == "Peak"].groupby(["peak", "substance"]).mean().sort_values(["rtDeviation"], ascending = False).reset_index()["substance"])
                for sub in dat["substance"]:
                    if sub not in orde:
                        orde.append(sub)
                p = (p9.ggplot(data = dat[dat["peak"] == "Peak"], mapping = p9.aes(x="rtDeviation", y="substance")) + p9.theme_minimal()
                    + p9.geom_jitter(alpha=0.2, width=0, height = 0.2)
                    + p9.scales.scale_y_discrete(limits=orde)
                    + p9.ggtitle("Rt deviation (min; apex - reference)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 32/250*len(ints), units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
            
                for sub in natsort.natsorted(list(ints), key = lambda x: x.lower()):
                    temp = dat[dat["peak"] == "Peak"]
                    temp = temp[temp["substance"] == sub]
                    a = temp.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.9])
                    body.append("<h3><p>%s</p></h3>"%(sub))
                    
                    if False:   ## TODO Problem with QFiledialogs.. disabled for now
                        if self.loadedExperiments[selExp].substances[sub].cas is not None:
                            body.append("<p>CAS: %s</p>"%(self.loadedExperiments[selExp].substances[sub].cas))
                        mol = None
                        if sub in self.loadedExperiments[selExp].substances:
                            if self.loadedExperiments[selExp].substances[sub].inchiKey is not None:
                                mol = Chem.MolFromInchi(self.loadedExperiments[selExp].substances[sub].inchiKey)
                                body.append("<p>Inchi: %s</p>"%(self.loadedExperiments[selExp].substances[sub].inchiKey))
                            elif self.loadedExperiments[selExp].substances[sub].canSmiles is not None:
                                mol = Chem.MolFromSmiles(self.loadedExperiments[selExp].substances[sub].canSmiles)
                                body.append("<p>Smiles%s</p>"%(self.loadedExperiments[selExp].substances[sub].canSmiles))
                            
                        if mol is not None:
                            Draw.MolToFile(mol, os.path.join(tmpDir, "tempFig.png"))
                            body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))                        
                    
                    body.append("<p>Note: Only chromatographic peaks (category 'Peak') are considered</p>")
                    body.append("<pre>%s</pre>"%(str(a).replace("\n", "<br>")))
                    
                    if False:
                        p = (p9.ggplot(data = temp[temp["substance"] == sub], mapping = p9.aes(x="peakWidth")) + p9.theme_minimal()
                            + p9.geom_histogram(bins = 100)
                            + p9.xlim([0, 2])
                            + p9.ggtitle("Peak width zoomed for '%s' (min)"%(sub)))
                        p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                        body.append("<img src='data:image/png;base64,%s'></img>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                        
                        p = (p9.ggplot(data = temp[temp["substance"] == sub], mapping = p9.aes(x="area")) + p9.theme_minimal()
                            + p9.geom_histogram(bins = 300)
                            + p9.scales.scale_x_log10()
                            + p9.ggtitle("Peak areas for '%s'"%(sub)))
                        p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                        body.append("<img src='data:image/png;base64,%s'></img>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                procDiag.close()

                outFil = os.path.join(tmpDir, "temp.html")
                with open(outFil, "w") as fout:
                    fout.write("%s%s%s"%(top, "\n".join(body), bot))
                                
                if False:
                    pass
                    ## TODO correct this bug; there is a bug with QFileDialog that does not show when the QWebEngineView is imported. 
                    #x = WebViewDialog(self, title = selExp, url = PyQt6.QtCore.QUrl("file:///%s"%(outFil.replace("\\", "/"))))
                    #x.setModal(True)
                    #x.setFixedWidth(1100)
                    #x.setFixedHeight(900)
                    #x.exec()
                else:
                    import webbrowser
                    webbrowser.open("file:///%s"%(outFil.replace("\\", "/")), new = 1)
                    PyQt6.QtWidgets.QMessageBox.information(self, "PeakBotMRM", "A summary of the results is shown in the webbrowser")

                
    def resortExperiments(self):
        
        sortMode = self.sortOrder.currentIndex()
        
        for c in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(c)
            selExp = it.experiment
            
            for subI in range(it.childCount()):
                subit = it.child(subI)
                selSub = subit.substance
                
                ints = []
                sampleNames = []
                for sampI in range(subit.childCount()):
                    samp = subit.child(sampI)
                    selSamp = samp.sample
                    ints.append(self.loadedExperiments[selExp].integrations[selSub][selSamp])
                    sampleNames.append(selSamp)
                
                while subit.childCount() > 0:
                    samp = subit.child(0)
                    subit.removeChild(samp)
                
                order = None
                if sortMode == 1:
                    order = [i[0] for i in sorted(enumerate(ints), key = lambda x: x[1].area if x[1].foundPeak % 128 else 0)]
                elif sortMode == 2:
                    order = [i[0] for i in sorted(enumerate(ints), key = lambda x: x[1].area if x[1].foundPeak % 128 else 0, reverse = True)]
                else:
                    temp = sortSamples(sampleNames, self.__defaultSampleOrder) 
                    order = []
                    for sampleName in temp:
                        order.append(sampleNames.index(sampleName))
                
                for i in order:
                    subit.addChild(ints[i].other["GUIElement"])
                    
    
    def resetActivateExperiment(self):
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "experiment" in l[0].__dict__:
            it = l[0]
            while it.parent() is not None:
                it = it.parent()
            
            selExp = it.experiment if "experiment" in it.__dict__ else None
            
            if selExp is not None and selExp != "" and selExp in self.loadedExperiments:
                
                button = PyQt6.QtWidgets.QMessageBox.question(self, "Reset experiment", "<b>Warning</b><br><br>This will reset the selected experiment (%s). Al progress (automated and manual annotations) will be lost. This action cannot be undone. <br>Are you sure that you want to continue?"%(selExp))
                if button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    
                    it.setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor1[0]), int(self.__highlightColor1[1]), int(self.__highlightColor1[2]), int(255 * 0.2)))
                    for s in self.loadedExperiments[selExp].integrations:
                        for h in self.loadedExperiments[selExp].integrations[s]:
                            self.loadedExperiments[selExp].integrations[s][h].foundPeak = 0
                            self.loadedExperiments[selExp].integrations[s][h].rtStart = -1
                            self.loadedExperiments[selExp].integrations[s][h].rtEnd = -1
                            self.loadedExperiments[selExp].integrations[s][h].area = -1
                    
                    for subNodeInd in range(it.childCount()):
                        subNode = it.child(subNodeInd)
                        for sampNodeInd in range(subNode.childCount()):
                            sampleItem = subNode.child(sampNodeInd)
                            inte = self.loadedExperiments[selExp].integrations[sampleItem.substance][sampleItem.sample]
                            sampleItem.setText(1, "")
                            sampleItem.setText(2, "")
                                
                    PyQt6.QtWidgets.QMessageBox.information(self, "Reset experiment", "Experiment '%s' has been reset. "%(selExp))
        else:
            PyQt6.QtWidgets.QMessageBox.warning(self, "PeakBotMRM", "<b>Warning</b><br><br>You cannot reset several experiments at once. Select one experiment and try again.")
    
    def exportIntegrations(self, all = False):
        ls = []
        outputFolder = None
        outputFile = None
        ext = ""
        preExt = ""
        if self.__exportSeparator == "\t": 
            ext = "Tab separated values files (*.tsv);;All files (*.*)"
            preExt = ".tsv"
        elif self.__exportSeparator in [",", ";", "$"]: 
            ext = "Comma separated values (*.csv);;All files (*.*)"
            preExt = ".csv"
        else:
            ext = "All files (*.*)"
            preExt = ".txt"
        
        if all:
            ls = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
            outputFolder = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        else:
            ls = list(self.tree.selectedItems())
            if len(ls) == 0:
                PyQt6.QtWidgets.QMessageBox.critical(self, "PeakBotMRM", "No experiment has been selected. Please select one from the list and retry.")
                return
            selExp = ls[0].experiment if "experiment" in ls[0].__dict__ else None
            
            fName = PyQt6.QtWidgets.QFileDialog.getSaveFileName(self, "Save results to file", directory = os.path.join(".", "%s_PB%s"%(selExp, preExt)), filter = ext, options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
            if fName[0]:
                outputFile = fName[0]
            else:
                return
        
        procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Exporting experiment(s)")
        procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
        procDiag.setWindowTitle("PeakBotMRM")
        procDiag.setModal(True)
        procDiag.setMaximum(len(ls))
        procDiag.setValue(0)
        procDiag.show()
        
        for i, it in enumerate(ls):
            procDiag.setValue(i)
            while it.parent() is not None:
                it = it.parent()
            
            selExp = it.experiment if "experiment" in it.__dict__ else None
            
            if outputFolder is not None:
                outputFile = os.path.join(outputFolder, "%s_PB%s"%(selExp, preExt))
            
            if selExp is not None and selExp != "" and selExp in self.loadedExperiments:
                substancesComments, samplesComments = PeakBotMRM.predict.calibrateIntegrations(self.loadedExperiments[selExp].substances, self.loadedExperiments[selExp].integrations)
                PeakBotMRM.predict.exportIntegrations(outputFile, 
                                                      self.loadedExperiments[selExp].substances, 
                                                      self.loadedExperiments[selExp].integrations, 
                                                      separator = self.__exportSeparator, 
                                                      substancesComments = substancesComments, 
                                                      samplesComments = samplesComments, 
                                                      sampleMetaData = self.loadedExperiments[selExp].sampleInfo, 
                                                      additionalCommentsForFile=[],
                                                      oneRowHeader4Results = False)
                if len(ls) == 1:
                    PyQt6.QtWidgets.QMessageBox.information(self, "Exporting results", "Experiment '%s' has been exported to file<br>'%s'"%(selExp, outputFile))
        
        procDiag.close()
        if len(ls) > 1:
            PyQt6.QtWidgets.QMessageBox.information(self, "Exporting results", "Experiment results have been exported")
    
    def exportReport(self):
        try:
            ext = ""
            preExt = ""
            if self.__exportSeparator == "\t": 
                ext = "Tab separated values files (*.tsv);;All files (*.*)"
                preExt = ".tsv"
            elif self.__exportSeparator in [",", ";", "$"]: 
                ext = "Comma separated values (*.csv);;All files (*.*)"
                preExt = ".csv"
            else:
                ext = "All files (*.*)"
                preExt = ".txt"
            
            ls = list(self.tree.selectedItems())
            if len(ls) != 1:
                PyQt6.QtWidgets.QMessageBox.critical(self, "PeakBotMRM", "No experiment has been selected. Please select one from the list and retry.")
                return
            
            selExp = ls[0].experiment
            
            fName = PyQt6.QtWidgets.QFileDialog.getSaveFileName(self, "Save results to file", directory = os.path.join(".", "%s_PBReport%s"%(selExp, preExt)), filter = ext, options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
            if fName[0]:
                with pyqtgraph.BusyCursor():
                    outputFile = fName[0]
                
                    PeakBotMRM.predict.calibrateIntegrations(self.loadedExperiments[selExp].substances, self.loadedExperiments[selExp].integrations)
                    
                    samples = set()
                    for sub in self.loadedExperiments[selExp].substances:
                        if self.loadedExperiments[selExp].substances[sub].type.lower() not in ["target", "istd"]:
                            raise Exception("Unknown substance type '%s' for '%s'"%(self.loadedExperiments[selExp].substances[sub].type, sub))
                        if sub in self.loadedExperiments[selExp].integrations:
                            for samp in self.loadedExperiments[selExp].integrations[sub]:
                                samples.add(samp)
                    
                    samples = sortSamples(list(samples), self.__defaultSampleOrder)
                    substances = natsort.natsorted([sub for sub in self.loadedExperiments[selExp].substances if self.loadedExperiments[selExp].substances[sub].type.lower() == "target"], key = lambda x: x.lower())
                    
                    with open(outputFile, "w", newline = "") as fout:
                        tsvWr = csv.writer(fout, delimiter = self.__exportSeparator)
                        tsvWr.writerow(["Sample", "Normalization", "Unit"] + substances)
                                
                        for samp in samples:
                            sampleType = self.loadedExperiments[selExp].sampleInfo[samp]["Type"]
                            if sampleType.lower() == "bio":
                                
                                sampleID = self.loadedExperiments[selExp].sampleInfo[samp]["Sample ID"]
                                if sampleID == "":
                                    raise Exception("Unknown sample type for sample '%s'"%(samp))
                                
                                reportDescription = self.loadedExperiments[selExp].sampleInfo[samp]["Report type"]
                                reportCalculation = self.loadedExperiments[selExp].sampleInfo[samp]["Report calculation"]
                                
                                units = set()
                                out = [str(sampleID), reportDescription + " (val = %s)"%(reportCalculation.split(",")[0]), reportCalculation.split(",")[1]]
                                for sub in substances:
                                    if sub in self.loadedExperiments[selExp].integrations and samp in self.loadedExperiments[selExp].integrations[sub] and self.loadedExperiments[selExp].integrations[sub][samp].concentration is not None:
                                        val = self.loadedExperiments[selExp].integrations[sub][samp].concentration[0]
                                        
                                        val, unit = PeakBotMRM.predict.calcNormalizedValue(val, self.loadedExperiments[selExp].sampleInfo[samp])
                                        units.add(unit)
                                        
                                        out.append(val)
                                    else:
                                        out.append("<LOQ")
                                if len(units) != 1:
                                    PyQt6.QtWidgets.QMessageBox.critical(self, "Exporting results", "<b>Error</b><br><br>Sample '%s' resulted in ambiguous units for the calculation.<br>The export will be aborted!"%(samp))
                                    logging.error("Error, several units calculated for one sample")
                                tsvWr.writerow(out)
                            
                        tsvWr.writerow(["#"])
                        tsvWr.writerow(["## General information"])
                        tsvWr.writerow(["## .. Date: '%s'"%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))])
                        tsvWr.writerow(["## .. Computer: '%s'"%(platform.node())])
                        tsvWr.writerow(["## Parameters"])
                        tsvWr.writerow(["## PeakBotMRM configuration"])
                        for r in PeakBotMRM.Config.getAsString().split(";"):
                            tsvWr.writerow(["## .. '" + r + "'"])
                
                PyQt6.QtWidgets.QMessageBox.information(self, "Exporting results", "Report for '%s' has been created and saved to file<br>'%s'"%(selExp, outputFile))
                
        except Exception as ex:
            PyQt6.QtWidgets.QMessageBox.critical(self, "Error", "An exception occured while generating the report.<br><br><b>%s</b>"%(ex))
            logging.exception("Exception")
        
                
    def closeExperiment(self):
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "experiment" in l[0].__dict__:
            it = l[0]
            while it.parent() is not None:
                it = it.parent()
            selExp = it.experiment if "experiment" in it.__dict__ else None
            
            if selExp is not None and selExp != "" and selExp in self.loadedExperiments:
                button = PyQt6.QtWidgets.QMessageBox.question(self, "Close experiment", "<b>Warning</b><br><br>This will close the selected experiment (%s). Any unsaved changes will be lost. This action cannot be undone. <br>Are you sure that you want to continue?"%(selExp))
                
                if button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))                    
                    del self.loadedExperiments[selExp]
                    
                    PyQt6.QtWidgets.QMessageBox.information(self, "Closed experiment", "Experiment '%s' has been closed."%(selExp))
        
    def genPlot(self):
        plot = pyqtgraph.PlotWidget()
        self._plots.append(plot)
        return plot
    
    def userSelectExperimentalData(self):
        dialog = OpenExperimentDialog(parent = self)
        dialog.setModal(True)
        okay = dialog.exec()
        if okay:
            expName, transitionsFile, rawDataPath, resultsFile, delimChar = dialog.getUserData()
            main.loadExperiment(expName, transitionsFile, rawDataPath, resultsFile, delimChar)
    
    def saveBinaryExperimentHelper(self):        
        ls = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        if len(ls) >= 1:
            
            fDir = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder for model and log", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
            if fDir:
                
                procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Saving experiments")
                procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
                procDiag.setWindowTitle("PeakBotMRM")
                procDiag.setModal(True)
                procDiag.setMaximum(len(ls))
                procDiag.show()
                
                with pyqtgraph.BusyCursor():
                    for i, l in enumerate(ls):
                        procDiag.setValue(i)
                        procDiag.setLabelText("Saving experiments, %d / %d done"%(i, len(ls)))
                        if not self.loadedExperiments[l.experiment].saveToFile(os.path.join(fDir, l.experiment+".pbexp"), additionalData = {"settings": copy.deepcopy(self._getSaveSettingsObject())}):
                            PyQt6.QtWidgets.QMessageBox.error(self, "PeakBotMRM", "<b>Error: could not export experiment</b><br><br>See log for further details.")
                        else:
                            procDiag.setLabelText("Saved experiment %s"%(l.experiment))
                        
                procDiag.close()
            PyQt6.QtWidgets.QMessageBox.information(self, "PeakBotMRM", "Experiment%s saved successfully"%("(s)" if len(ls) > 1 else "'%s'"%ls[0].experiment))    
        else:
            PyQt6.QtWidgets.QMessageBox.warning(self, "PeakBotMRM", "Please select an experiment from the list first")        
        
    def loadBinaryExperimentHelper(self):
        files = PyQt6.QtWidgets.QFileDialog.getOpenFileNames(self, "Open binary experiment", filter="PeakBotMRM experiments (*.pbexp)", options = PyQt6.QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if len(files[0]) > 0:
            
            procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Loading experiments")
            procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
            procDiag.setWindowTitle("PeakBotMRM")
            procDiag.setModal(True)
            procDiag.setMaximum(len(files[0]))
            procDiag.show()
            
            with pyqtgraph.BusyCursor():
                for i, file in enumerate(files[0]):
                    procDiag.setValue(i)
                    procDiag.setLabelText("Loading experiments, %d / %d done"%(i, len(files[0])))
                    self.loadBinaryExperiment(file)
                
            procDiag.close()

    def loadBinaryExperiment(self, fromFile):
        with open(fromFile, "rb") as fin:
            expName, substances, integrations, sampleInfo, additionalData = pickle.load(fin)
            
            i = 1
            while expName in self.loadedExperiments:
                expName = expName + "_" + str(i)
                i = i + 1
            
            if additionalData is not None and type(additionalData) == dict and "settings" in additionalData:
                self._loadSettingsFromObject(additionalData["settings"])
            
            self.addExperimentToGUI(expName, substances, integrations, sampleInfo)
    
    def loadExperiment(self, expName, transitionFile, rawDataPath, integrationsFile, delimChar):
                
        procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Loading experiment '%s'"%(expName))
        procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
        procDiag.setWindowTitle("PeakBotMRM")
        procDiag.setModal(True)
        procDiag.show()
        
        substances = PeakBotMRM.loadTargets(transitionFile, 
                                            logPrefix = "  | ..")
        
        integrations = None
        integrationsLoaded = False
        with pyqtgraph.BusyCursor():
            if integrationsFile is not None and integrationsFile != "":
                substances, integrations = PeakBotMRM.loadIntegrations(substances, 
                                                                    integrationsFile, 
                                                                    delimiter = delimChar, 
                                                                    logPrefix = "  | ..")
                integrationsLoaded = True
            
            substances, integrations, sampleInfo = PeakBotMRM.loadChromatograms(substances, integrations, 
                                                                                rawDataPath, 
                                                                                pathToMSConvert = self.__msConvertPath, 
                                                                                maxValCallback = procDiag.setMaximum, curValCallback = procDiag.setValue, 
                                                                                logPrefix = "  | ..",
                                                                                errorCallback = functools.partial(TimerMessageBox, self, "File failed", timeout = 10))
        
        for substance in substances:
            if substances[substance].type.lower() not in ["target", "istd"]:
                PyQt6.QtWidgets.QMessageBox.ciritical(self, "Error with substance type", "Error<br><br>Substance type '%s' for '%s' not valid.<br>Must be  'Target' or 'ISTD.<br><br>Import will be aborted"%(substance.type, substance.name))
                return
        
        procDiag.close()       
        
        self.addExperimentToGUI(expName, substances, integrations, sampleInfo, integrationsLoaded)

    def addExperimentToGUI(self, expName, substances, integrations, sampleInfo, integrationsLoaded = False, showProcDiag = True):
        self.tree.blockSignals(True)
        self.loadedExperiments[expName] = Experiment(expName, substances, integrations, sampleInfo)
        
        if showProcDiag:
            procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Loading experiment '%s'"%(expName))
            procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
            procDiag.setWindowTitle("PeakBotMRM")
            procDiag.setModal(True)
            procDiag.setMaximum(len(self.loadedExperiments[expName].sampleInfo))
            procDiag.show()
        
        i = 0
        for k, v in self.loadedExperiments[expName].sampleInfo.items():
            if showProcDiag:
                procDiag.setValue(i)
            i = i + 1
            
            if "Group" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Group"]     = ""
            if "Type" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Type"]      = ""
            if "Color" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Color"]     = ""
            if "use4Stats" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["use4Stats"] = False
            if "Dilution" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Dilution"] = ""
            if "Inj. volume" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Inj. volume"] = ""
            if "Sample ID" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Sample ID"] = ""
            if "Report type" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Report type"] = ""
            if "Cell count" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Cell count"] = ""
            if "Tissue weight" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Tissue weight"] = ""
            if "Sample volume" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Sample volume"] = ""
            if "Report description" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Report description"] = "None"
            if "Report calculation" not in self.loadedExperiments[expName].sampleInfo[k]:
                self.loadedExperiments[expName].sampleInfo[k]["Report calculation"] = "val"
                       
        for k, v in self.loadedExperiments[expName].sampleInfo.items():
            self.loadedExperiments[expName].sampleInfo[k]["Name"] = Path(self.loadedExperiments[expName].sampleInfo[k]["path"]).stem
            self.loadedExperiments[expName].sampleInfo[k]["Data File"] = os.path.basename(self.loadedExperiments[expName].sampleInfo[k]["path"])
            ## TODO add other info here
        
        rootItem = PyQt6.QtWidgets.QTreeWidgetItem(self.tree)
        rootItem.setText(0, expName)
        if not integrationsLoaded:
            rootItem.setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor1[0]), int(self.__highlightColor1[1]), int(self.__highlightColor1[2]), int(255 * 0.2)))
        rootItem.experiment = expName; rootItem.substance = None; rootItem.sample = None
        
        if True:
            allSamples = []
            curi = 0
            for substance in natsort.natsorted(integrations, key = lambda x: x.lower()):
                procDiag.setValue(curi)
                curi = curi + 1
                
                substanceItem = PyQt6.QtWidgets.QTreeWidgetItem(rootItem)
                substanceItem.experiment = expName; substanceItem.substance = substance; substanceItem.sample = None; substanceItem.userType = "All samples"
                substanceItem.setText(0, substance)
                if substance in self.loadedExperiments[expName].substances:
                    s = self.loadedExperiments[expName].substances[substance].internalStandard
                    if s is not None and s != "":
                        substanceItem.setText(1, s)
                else:
                    substanceItem.setText(1, "not found in transition list")
                
                for sample in sortSamples(integrations[substance], self.__defaultSampleOrder):
                    inte = integrations[substance][sample]
                    sampleItem = PyQt6.QtWidgets.QTreeWidgetItem(substanceItem)
                    sampleItem.experiment = expName; sampleItem.substance = substance; sampleItem.sample = sample; sampleItem.userType = "Single peak"
                    showName = sample
                    for temp, rep in self.__sampleNameReplacements.items():
                        showName = showName.replace(temp, rep)
                    sampleItem.setText(0, showName)
                    if inte.foundPeak != None:
                        sampleItem.setIcon(0, {0: self.__icons["res/PB/nothing"], 1: self.__icons["res/PB/peak"], 2: self.__icons["res/PB/noise"], 128: self.__icons["res/manual/nothing"], 129: self.__icons["res/manual/peak"], 130: self.__icons["res/manual/noise"]}[inte.foundPeak])
                        sampleItem.setText(1, self.__areaFormatter%(inte.area) if inte.foundPeak % 128 else "")
                        sampleItem.setText(2, "%.2f - %.2f"%(inte.rtStart, inte.rtEnd) if inte.foundPeak % 128 != 0 else "")
                    inte.other["GUIElement"] = sampleItem
                    
                    allSamples.append(sample)
        
        if showProcDiag:
            procDiag.close()        
        self.tree.blockSignals(False)
        
    def curInterpolationFunctionChanged(self):
        self.tree.blockSignals(True); self.hasPeak.blockSignals(True); self.peakStart.blockSignals(True); self.peakEnd.blockSignals(True); self.istdhasPeak.blockSignals(True); self.istdpeakStart.blockSignals(True); self.istdpeakEnd.blockSignals(True); self.useForCalibration.blockSignals(True); self.calibrationMethod.blockSignals(True);
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "userType" in l[0].__dict__:
            it = l[0]
            selExp = it.experiment if "experiment" in it.__dict__ else None
            selSub = it.substance if "substance" in it.__dict__ else None
            if selExp is not None and selSub is not None:
                self.loadedExperiments[selExp].substances[selSub].calibrationMethod = self.calibrationMethod.currentText()
                self.featurePropertiesChanged()
        
        self.tree.blockSignals(False); self.hasPeak.blockSignals(False); self.peakStart.blockSignals(False); self.peakEnd.blockSignals(False); self.istdhasPeak.blockSignals(False); self.istdpeakStart.blockSignals(False); self.istdpeakEnd.blockSignals(False); self.useForCalibration.blockSignals(False); self.calibrationMethod.blockSignals(False);
    
    def featurePropertiesChanged(self):
        with pyqtgraph.BusyCursor():
            self.tree.blockSignals(True); self.hasPeak.blockSignals(True); self.peakStart.blockSignals(True); self.peakEnd.blockSignals(True); self.istdhasPeak.blockSignals(True); self.istdpeakStart.blockSignals(True); self.istdpeakEnd.blockSignals(True); self.useForCalibration.blockSignals(True); self.calibrationMethod.blockSignals(True);
            its = list(self.tree.selectedItems())
            for it in its:
                if "userType" in it.__dict__ and it.userType == "Single peak":
                    selExp = it.experiment if "experiment" in it.__dict__ else None
                    selSub = it.substance if "substance" in it.__dict__ else None
                    selSam = it.sample if "sample" in it.__dict__ else None
                    selIST = self.loadedExperiments[selExp].substances[selSub].internalStandard if selExp is not None and selSub is not None and self.loadedExperiments[selExp].substances[selSub].internalStandard is not None and self.loadedExperiments[selExp].substances[selSub].internalStandard != "" else None
                    
                    if selExp is not None and selSub is not None:
                        self.calibrationMethod.setCurrentIndex(self.calibrationMethod.findText(self.loadedExperiments[selExp].substances[selSub].calibrationMethod))
                    
                    if "_CAL" in selSam:
                        m = re.search("(_CAL[0-9]+_)", selSam)
                        x = m.group(0)
                        if x in self.loadedExperiments[selExp].substances[selSub].calSamples:
                            self.loadedExperiments[selExp].substances[selSub].calSamples[x] = abs(self.loadedExperiments[selExp].substances[selSub].calSamples[x]) * (1 if self.useForCalibration.isChecked() else -1)
                    ## TODO peak
                    inte = self.loadedExperiments[selExp].integrations[selSub][selSam]
                    old = inte.foundPeak
                    inte.foundPeak = {0:0, 1:1, 2:2, 3:128, 4:129, 5:130}[self.hasPeak.currentIndex()] % 128 + 128
                    peakSwitched = old%128 == 0 and inte.foundPeak%128 > 0
                    it.setIcon(0, {0: self.__icons["res/PB/nothing"], 1: self.__icons["res/PB/peak"], 2: self.__icons["res/PB/noise"], 128: self.__icons["res/manual/nothing"], 129: self.__icons["res/manual/peak"], 130: self.__icons["res/manual/noise"]}[inte.foundPeak])
                    
                    if inte.foundPeak % 128:
                        eic = inte.chromatogram["eic"]
                        rts = inte.chromatogram["rts"]
                        
                        if peakSwitched:
                            self.peakStart.setValue(self.loadedExperiments[selExp].substances[selSub].refRT + self.__leftPeakDefault)
                            self.peakEnd.setValue(self.loadedExperiments[selExp].substances[selSub].refRT + self.__rightPeakDefault)
                        
                        inte.rtStart = self.peakStart.value()
                        inte.rtEnd = self.peakEnd.value()
                        
                        if PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT and peakSwitched:
                            startInd = PeakBotMRM.core.arg_find_nearest(rts, inte.rtStart)
                            endInd   = PeakBotMRM.core.arg_find_nearest(rts, inte.rtEnd)
                            while startInd + 1 < eic.shape[0] and eic[startInd + 1] >= eic[startInd]:
                                startInd = startInd + 1
                            while startInd - 1 >= 0 and eic[startInd - 1] <= eic[startInd]:
                                startInd = startInd - 1
                            while endInd + 1 < eic.shape[0] and eic[endInd + 1] <= eic[endInd]:
                                endInd = endInd + 1
                            
                            inte.rtStart = rts[startInd]
                            inte.rtEnd = rts[endInd]
                            self.peakStart.setValue(inte.rtStart)
                            self.peakEnd.setValue(inte.rtEnd)
                        
                        inte.area = PeakBotMRM.integrateArea(inte.chromatogram["eic"], inte.chromatogram["rts"], inte.rtStart, inte.rtEnd)
                        it.setText(1, self.__areaFormatter%(inte.area))
                        it.setText(2, "%.2f - %.2f"%(inte.rtStart, inte.rtEnd) if inte.foundPeak else "")
                    else:
                        it.setText(1, "")
                        it.setText(2, "")
                    inte.type = "Manual integration"
                    inte.comment = ""
                    inte.other["GUIElement"].setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor1[0]), int(self.__highlightColor1[1]), int(self.__highlightColor1[2]), int(255 * 0.2)))
                        
                    if selIST is not None and selIST in self.loadedExperiments[selExp].substances and selIST in self.loadedExperiments[selExp].integrations and selSam in self.loadedExperiments[selExp].integrations[selIST]:
                        inte = self.loadedExperiments[selExp].integrations[selIST][selSam]
                        old = inte.foundPeak
                        inte.foundPeak = {0:0, 1:1, 2:2, 3:128, 4:129, 5:130}[self.istdhasPeak.currentIndex()] % 128 + 128
                        peakSwitched = old%128 == 0 and inte.foundPeak%128 > 0
                        inte.other["GUIElement"].setIcon(0, {0: self.__icons["res/PB/nothing"], 1: self.__icons["res/PB/peak"], 2: self.__icons["res/PB/noise"], 128: self.__icons["res/manual/nothing"], 129: self.__icons["res/manual/peak"], 130: self.__icons["res/manual/noise"]}[inte.foundPeak])

                        if inte.foundPeak:
                            eic = inte.chromatogram["eic"]
                            rts = inte.chromatogram["rts"]
                            
                            if peakSwitched:
                                self.istdpeakStart.setValue(self.loadedExperiments[selExp].substances[selIST].refRT - 0.25)
                                self.istdpeakEnd.setValue(self.loadedExperiments[selExp].substances[selIST].refRT + 0.25)
                                
                            inte.rtStart = self.istdpeakStart.value()
                            inte.rtEnd = self.istdpeakEnd.value()
                        
                            if PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT and peakSwitched:
                                startInd = PeakBotMRM.core.arg_find_nearest(rts, inte.rtStart)
                                endInd   = PeakBotMRM.core.arg_find_nearest(rts, inte.rtEnd)
                                while startInd + 1 < eic.shape[0] and eic[startInd + 1] >= eic[startInd]:
                                    startInd = startInd + 1
                                while startInd - 1 >= 0 and eic[startInd - 1] <= eic[startInd]:
                                    startInd = startInd - 1
                                while endInd + 1 < eic.shape[0] and eic[endInd + 1] <= eic[endInd]:
                                    endInd = endInd + 1
                                
                                inte.rtStart = rts[startInd]
                                inte.rtEnd = rts[endInd]
                                self.istdpeakStart.setValue(inte.rtStart)
                                self.istdpeakEnd.setValue(inte.rtEnd)
                                
                            inte.area = PeakBotMRM.integrateArea(inte.chromatogram["eic"], inte.chromatogram["rts"], inte.rtStart, inte.rtEnd)
                            inte.other["GUIElement"].setText(1, self.__areaFormatter%(inte.area))
                            inte.other["GUIElement"].setText(2, "%.2f - %.2f"%(inte.rtStart, inte.rtEnd) if inte.foundPeak else "")
                        else:
                            inte.other["GUIElement"].setText(1, "")
                            inte.other["GUIElement"].setText(2, "")
                            
                        inte.type = "Manual integration"
                        inte.comment = ""            
                        inte.other["GUIElement"].setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor1[0]), int(self.__highlightColor1[1]), int(self.__highlightColor1[2]), int(255 * 0.2)))
                
            self.treeSelectionChanged()
            self.tree.blockSignals(False); self.hasPeak.blockSignals(False); self.peakStart.blockSignals(False); self.peakEnd.blockSignals(False); self.istdhasPeak.blockSignals(False); self.istdpeakStart.blockSignals(False); self.istdpeakEnd.blockSignals(False);  self.useForCalibration.blockSignals(False); self.calibrationMethod.blockSignals(False);
    
    def refreshViews(self, autoRange = True):
        self.lastExp = None
        self.lastSam = None
        self.lastSub = None
        
        if autoRange:
            for plot in self._plots:
                plot.enableAutoRange()
        
        self.featurePropertiesChanged()
    
    def treeSelectionChanged(self):
        with pyqtgraph.BusyCursor():
            its = list(self.tree.selectedItems())
            self.tree.blockSignals(True); self.hasPeak.blockSignals(True); self.peakStart.blockSignals(True); self.peakEnd.blockSignals(True); self.istdhasPeak.blockSignals(True); self.istdpeakStart.blockSignals(True); self.istdpeakEnd.blockSignals(True); self.useForCalibration.blockSignals(True); self.calibrationMethod.blockSignals(True);
            if len(its) > 0:
                selExps = set()
                selSubs = set()
                selSams = set()
                selISTs = set()
                
                for it in its:
                    selExp = it.experiment if "experiment" in it.__dict__ else None
                    selSub = it.substance if "substance" in it.__dict__ else None
                    selSam = it.sample if "sample" in it.__dict__ else None
                    selIST = self.loadedExperiments[selExp].substances[selSub].internalStandard if selExp is not None and selSub is not None and self.loadedExperiments[selExp].substances[selSub].internalStandard is not None and self.loadedExperiments[selExp].substances[selSub].internalStandard != "" else None
                    selExps.add(selExp)
                    selSubs.add(selSub)
                    selSams.add(selSam)
                    selISTs.add(selIST)
                
                if len(selExps) > 1 or len(selSubs) > 1 or len(selISTs) > 1 or (len(selSams)>1 and None in selSams):
                    PyQt6.QtWidgets.QMessageBox.warning(self, "PeakBotMRM", "<b>Warning</b><br><br>Selecting several different experiments or substances is not supported at this time.<br>Please only select different samples if necessary!")
                
                else:                
                    if selExp == self.lastExp and selSub != self.lastSub:
                        for plot in self._plots:
                            plot.enableAutoRange()
                    
                    for i, plot in enumerate(self._plots):
                        if i in [1,2,4,5,9] and selExp == self.lastExp and selSub == self.lastSub:
                            pass
                        else:
                            plot.clear()
                    
                    self.hasPeak.setCurrentIndex(0); self.peakStart.setValue(0); self.peakEnd.setValue(0); self.istdhasPeak.setCurrentIndex(0); self.istdpeakStart.setValue(0); self.istdpeakEnd.setValue(0)
                    
                    self.infoLabel.setText("Selected: Experiment <b>%s</b>, Substance <b>%s</b> (ISTD <b>%s</b>), Sample(s) <b>%s</b>"%(
                        selExp if selExp is not None else "-",
                        selSub if selSub is not None else "-",
                        selIST if selIST is not None else "-",
                        (str(len(selSams)) if len(selSams) > 1 else selSam) if selSam is not None else "-"
                    ))
                    it = None
                    for it in its:
                        if "userType" in it.__dict__:
                            if it.userType == "Single peak":
                                selSam = it.sample if "sample" in it.__dict__ else None
                                
                                inte = self.loadedExperiments[selExp].integrations[selSub][selSam]
                                if len(its) == 1:
                                        self.infoLabel.setText("%s<br>Sub integration <b>%s%s</b>"%(self.infoLabel.text(), inte.type, " (%s)"%(inte.comment) if inte.comment is not None and inte.comment != "" else ""))
                                isCalSample = False
                                for sampPart, level in self.loadedExperiments[selExp].substances[selSub].calSamples.items():
                                    if sampPart in selSam:
                                        self.useForCalibration.setChecked(level > 0)
                                        isCalSample = True
                                self.useForCalibration.setEnabled(isCalSample)
                                if not isCalSample:
                                    self.useForCalibration.setChecked(False)
                                
                                if inte.foundPeak is not None and inte.foundPeak % 128:
                                    self.hasPeak.setCurrentIndex({0:0, 1:1, 2:2, 128:3, 129:4, 130:5}[inte.foundPeak])
                                    self.peakStart.setValue(inte.rtStart)
                                    self.peakEnd.setValue(inte.rtEnd)
                                self.plotIntegration(inte, "Sub", refRT = self.loadedExperiments[selExp].substances[selSub].refRT, plotInd = 0, transp = max(0.05, 0.8 /len(its)))
                                
                                if selIST is not None and selIST in self.loadedExperiments[selExp].substances and selIST in self.loadedExperiments[selExp].integrations and selSam in self.loadedExperiments[selExp].integrations[selIST]:
                                    inte = self.loadedExperiments[selExp].integrations[selIST][selSam]
                                    if len(its) == 1:
                                        self.infoLabel.setText("%s<br>ISTD integration <b>%s%s</b>"%(self.infoLabel.text(), inte.type, " (%s)"%(inte.comment) if inte.comment is not None and inte.comment != "" else ""))
                                    
                                    if inte.foundPeak is not None and inte.foundPeak % 128:
                                        self.istdhasPeak.setCurrentIndex({0:0, 1:1, 2:2, 128:3, 129:4, 130:5}[inte.foundPeak])
                                        self.istdpeakStart.setValue(inte.rtStart)
                                        self.istdpeakEnd.setValue(inte.rtEnd)
                                    self.plotIntegration(inte, "ISTD", refRT = self.loadedExperiments[selExp].substances[selIST].refRT if selIST is not None and selIST in self.loadedExperiments[selExp].substances else None, plotInd = 3, transp = max(0.05, 0.8 /len(its)))
                                it = it.parent()
                        
                    if selExp != self.lastExp or selSub != self.lastSub:
                        if "userType" in it.__dict__:
                            if it.userType == "All samples":
                                ints = []
                                intsIS = []
                                colors = []
                                colorsIS = []
                                for oitInd in range(it.childCount()):
                                    oit = it.child(oitInd)
                                    if "userType" in oit.__dict__ and oit.userType == "Single peak" and not oit.isHidden():
                                        ints.append(self.loadedExperiments[oit.experiment].integrations[oit.substance][oit.sample])
                                        colors.append(self.loadedExperiments[oit.experiment].sampleInfo[oit.sample]["Color"])
                                    if oit.experiment in self.loadedExperiments and oit.substance in self.loadedExperiments[oit.experiment].substances and self.loadedExperiments[oit.experiment].substances[oit.substance].internalStandard in self.loadedExperiments[oit.experiment].integrations and oit.sample in self.loadedExperiments[oit.experiment].integrations[self.loadedExperiments[oit.experiment].substances[oit.substance].internalStandard] and not oit.isHidden():
                                        intsIS.append(self.loadedExperiments[oit.experiment].integrations[self.loadedExperiments[oit.experiment].substances[oit.substance].internalStandard][oit.sample])
                                        colorsIS.append(self.loadedExperiments[oit.experiment].sampleInfo[oit.sample]["Color"])
                                if len(ints) > 0:
                                    self.plotIntegrations(ints, colors, "All EICs Sub", refRT = self.loadedExperiments[selExp].substances[selSub].refRT, plotInds = [1,2])
                                if len(intsIS) > 0:
                                    self.plotIntegrations(intsIS, colorsIS, "All EICs ISTD", refRT = self.loadedExperiments[selExp].substances[selIST].refRT if selIST is not None and selIST in self.loadedExperiments[selExp].substances else None, plotInds = [4,5])
                            
                    if "userType" in it.__dict__:
                        if it.userType == "All samples" and it.substance is not None and it.substance in self.loadedExperiments[it.experiment].substances:
                            substance = self.loadedExperiments[selExp].substances[selSub]
                            istd = self.loadedExperiments[selExp].substances[substance.internalStandard] if substance.internalStandard is not None and substance.internalStandard != "" and substance.internalStandard in self.loadedExperiments[selExp].substances else None
                            
                            subArea = []
                            isArea = []
                            subRatio = []
                            
                            highlightLevel = None
                            highlightLevelIS = None
                            highlightLevelRatio = None
                            for oitInd in range(it.childCount()):
                                oit = it.child(oitInd)
                                for calSampPart, level in substance.calSamples.items():
                                    if "userType" in oit.__dict__ and oit.userType == "Single peak" and calSampPart in oit.sample:
                                        if self.loadedExperiments[oit.experiment].integrations[oit.substance][oit.sample].foundPeak is not None and self.loadedExperiments[oit.experiment].integrations[oit.substance][oit.sample].foundPeak % 128 == 1:
                                            subArea.append((self.loadedExperiments[oit.experiment].integrations[oit.substance][oit.sample].area, level * substance.calLevel1Concentration))
                                            if oit.sample == selSam:
                                                highlightLevel = level * substance.calLevel1Concentration
                                        if istd is not None:
                                            if self.loadedExperiments[oit.experiment].integrations[istd.name][oit.sample].foundPeak is not None and self.loadedExperiments[oit.experiment].integrations[istd.name][oit.sample].foundPeak % 128 == 1:
                                                isArea.append((self.loadedExperiments[oit.experiment].integrations[istd.name][oit.sample].area, level * istd.calLevel1Concentration))
                                                if oit.sample == selSam:
                                                    highlightLevelIS = level
                                                if self.loadedExperiments[oit.experiment].integrations[oit.substance][oit.sample].foundPeak % 128 == 1 and self.loadedExperiments[oit.experiment].integrations[istd.name][oit.sample].foundPeak % 128 == 1: 
                                                    subRatio.append((self.loadedExperiments[oit.experiment].integrations[oit.substance][oit.sample].area / self.loadedExperiments[oit.experiment].integrations[istd.name][oit.sample].area, level * substance.calLevel1Concentration))
                                                    if oit.sample == selSam:
                                                        highlightLevelRatio = level * substance.calLevel1Concentration
                            self.calibrationMethod.setCurrentIndex(self.calibrationMethod.findText(self.loadedExperiments[selExp].substances[selSub].calibrationMethod))
                            
                            if len(subArea) > 0:
                                self.plotCalibration(subArea, "Sub; areas", unit = substance.calLevel1ConcentrationUnit, addLM = True, highlightLevel = highlightLevel, plotInd = 6)
                            if len(isArea) > 0:
                                self.plotCalibration(isArea, "ISTD, areas", unit = substance.calLevel1ConcentrationUnit, addLM = False, highlightLevel = highlightLevelIS, plotInd = 7)
                            if len(subRatio) > 0:
                                self.plotCalibration(subRatio, "Sub/ISTD", unit = substance.calLevel1ConcentrationUnit, addLM = True, highlightLevel = highlightLevelRatio, plotInd = 8)
                    
                    
                    if selExp != self.lastExp or self.paCMAP is None:
                        self.paCMAPAllEICsS = []
                        self.paCMAPAllRTsS = []
                        self.paCMAPsubstances = []
                        self.paCMAPSamples = []
                        self.paCMAP = None
                        self.polyROI = None
                        for sub in self.loadedExperiments[selExp].integrations:
                            for samp in self.loadedExperiments[selExp].integrations[sub]:
                                inte = self.loadedExperiments[selExp].integrations[sub][samp]
                                if "eicStd" not in inte.chromatogram:
                                    rtsS, eicS = PeakBotMRM.extractStandardizedEIC(inte.chromatogram["eic"], inte.chromatogram["rts"], self.loadedExperiments[selExp].substances[sub].refRT)
                                    inte.chromatogram["eicS"] = eicS
                                    inte.chromatogram["rtsS"] = rtsS
                                a = inte.chromatogram["eicS"]
                                a = a - np.min(a)
                                a = a / np.max(a)
                                self.paCMAPAllEICsS.append(a)
                                self.paCMAPAllRTsS.append(inte.chromatogram["rtsS"])
                                self.paCMAPsubstances.append(sub)
                                self.paCMAPSamples.append(samp)
                                
                        # loading preprocessed coil_20 dataset
                        # you can change it with any dataset that is in the ndarray format, with the shape (N, D)
                        # where N is the number of samples and D is the dimension of each sample
                        # initializing the pacmap instance
                        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
                        embedding = pacmap.PaCMAP(n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 

                        # fit the data (The index of transformed data corresponds to the index of the original data)
                        ## TODO: Dialog is not rendered correctly, only a blank window appears
                        procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Generating PaCMAP embedding", 
                                                                    minimum=0, maximum=1)
                        procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
                        procDiag.setWindowTitle("PeakBotMRM")
                        procDiag.setModal(True)
                        procDiag.show()
                        self.paCMAP = embedding.fit_transform(np.array(self.paCMAPAllEICsS), init="pca")
                        procDiag.close()
                    
                    if self.paCMAP is not None and (selExp != self.lastExp or selSam != self.lastSam or selSub != self.lastSub):            
                        self.polyROI = None
                        self.plotPaCMAP(selSub, selSam)
                    
                    self.lastExp = selExp
                    self.lastSub = selSub
                    self.lastSam = selSam
            
            self.tree.blockSignals(False); self.hasPeak.blockSignals(False); self.peakStart.blockSignals(False); self.peakEnd.blockSignals(False); self.istdhasPeak.blockSignals(False); self.istdpeakStart.blockSignals(False); self.istdpeakEnd.blockSignals(False); self.useForCalibration.blockSignals(False); self.calibrationMethod.blockSignals(False);

    def plotPaCMAP(self, selSub, selSam, addROI = False):
        with pyqtgraph.BusyCursor():
            def points_in_polygon(polygon, pts):
                polygon = np.asarray(polygon,dtype='float32')
                
                pts = np.asarray(pts,dtype='float32')
                
                contour2 = np.vstack((polygon[1:], polygon[:1]))
                test_diff = contour2-polygon
                mask1 = (pts[:,None] == polygon).all(-1).any(-1)
                m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
                slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
                m2 = slope == 0
                mask2 = (m1 & m2).any(-1)
                m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
                m4 = m1 & m3
                count = np.count_nonzero(m4,axis=-1)
                mask3 = ~(count%2==0)
                mask = mask1 | mask2 | mask3
                return mask
            # visualize the embedding
            self._plots[9].clear()
            self._plots[9].plot(self.paCMAP[:, 0], self.paCMAP[:, 1], pen=None, symbol='o', symbolPen=(self.__normalColor[0], self.__normalColor[1], self.__normalColor[2], int(255*0.66)), symbolSize=4, symbolBrush=None)
            
            if self.polyROI is not None:
                poly = np.array([(q[1].x(), q[1].y()) for q in self.polyROI.getLocalHandlePositions()], dtype="float32")
                points = self.paCMAP
                
                self.polyROI = pyqtgraph.PolyLineROI(poly, closed=True, pen=self.__highlightColor1)
                self.polyROI.sigRegionChangeFinished.connect(self.updatePACMAPROI)
                self._plots[9].addItem(self.polyROI)

                mask = points_in_polygon(poly, points)
                self._plots[9].plot(self.paCMAP[mask, 0], self.paCMAP[mask, 1], pen=None, symbol='o', symbolPen="Orange", symbolSize=8, symbolBrush="Orange")
                
                ints = []
                colors = []
                for xi, m in enumerate(mask):            
                    if m:
                        xsub = self.paCMAPsubstances[xi]
                        xsam = self.paCMAPSamples[xi]
                        ints.append(self.loadedExperiments[self.lastExp].integrations[xsub][xsam])
                        colors.append(self.loadedExperiments[self.lastExp].sampleInfo[xsam]["Color"])
                        
                if len(ints) > 0:
                    self._plots[1].clear()
                    self._plots[2].clear()
                    self.plotIntegrations(ints, colors, "All EICs Sub", plotInds = [1,2], makeUniformRT = True, scaleEIC = True)
            elif addROI:
                self.polyROI = pyqtgraph.PolyLineROI([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]], closed=True, pen=self.__highlightColor1)
                self.polyROI.sigRegionChangeFinished.connect(self.updatePACMAPROI)
                self._plots[9].addItem(self.polyROI)
            
            highlight = []
            highlightSingle = None
            for i in range(len(self.paCMAPsubstances)):
                if self.paCMAPsubstances[i] == selSub and self.paCMAPSamples[i] == selSam:
                    highlightSingle = i
                    
                if self.paCMAPsubstances[i] == selSub:
                    highlight.append(i)
            if len(highlight) > 0:
                self._plots[9].plot(self.paCMAP[highlight, 0], self.paCMAP[highlight, 1], pen=None, symbol='o', symbolPen=(0, self.__highlightColor1[1], self.__highlightColor1[2], int(255*1)), symbolSize=4, symbolBrush=(0, self.__highlightColor1[1], self.__highlightColor1[2], int(255*1)))
            if highlightSingle is not None:
                self._plots[9].plot(self.paCMAP[[highlightSingle], 0], self.paCMAP[[highlightSingle], 1], pen=None, symbol='o', symbolPen=(self.__highlightColor1[0], self.__highlightColor1[1], self.__highlightColor1[2], int(255*1)), symbolSize=4, symbolBrush=(self.__highlightColor1[0], self.__highlightColor1[1], self.__highlightColor1[2], int(255*1)))
            
    def updatePACMAPROI(self):
        self.plotPaCMAP(self.lastSub, self.lastSam)
        
    def addPolyROItoPaCMAP(self):
        self.polyROI = None
        self.plotPaCMAP(self.lastSub, self.lastSam, addROI = True)
    
    def plotIntegration(self, inte, title, refRT = None, plotInd = 0, transp = 0.2):
        self._plots[plotInd].plot(inte.chromatogram["rts"], inte.chromatogram["eic"], pen = self.__normalColor)
        
        if inte.foundPeak is not None and inte.foundPeak % 128:
            try:
                rts = inte.chromatogram["rts"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                eic = inte.chromatogram["eic"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                pen = self.__highlightColor1
                if inte.foundPeak % 128 == 2:
                    pen = self.__highlightColor2
                p = self._plots[plotInd].plot(rts, eic, pen = pen)
                a = np.min(eic)
                p1 = self._plots[plotInd].plot(rts, np.ones(rts.shape[0]) * a)
                brush = (self.__highlightColor1[0], self.__highlightColor1[1], self.__highlightColor1[2], 255*transp)
                if inte.foundPeak % 128 == 2:
                    brush = (self.__highlightColor2[0], self.__highlightColor2[1], self.__highlightColor2[2], 255*transp)
                fill = pyqtgraph.FillBetweenItem(p, p1, brush)
                fill.setZValue(-1)
                self._plots[plotInd].addItem(fill)
            except:
                pass
            
        if refRT is not None:
            infLine = pyqtgraph.InfiniteLine(pos = [refRT, 0], movable=False, angle=90, label='', pen=self.__normalColor)
            self._plots[plotInd].addItem(infLine)            
            
        self._plots[plotInd].setTitle(title)
        self._plots[plotInd].setLabel('left', "Intensity")
        self._plots[plotInd].setLabel('bottom', "Retention time (min)")
        
    def plotIntegrations(self, intes, colors, title, refRT = None, plotInds = [1,2], makeUniformRT = False, scaleEIC = False):
        
        for ind in range(len(intes)):
            inte = intes[ind]
            x = inte.chromatogram["rts"]
            y = inte.chromatogram["eic"]
            if makeUniformRT:
                x = np.linspace(0, 1, x.shape[0])
            if scaleEIC:
                y = y - np.min(y)
                y = y / np.max(y)
            col = PyQt6.QtGui.QColor.fromRgb(self.__normalColor[0], self.__normalColor[1], self.__normalColor[2]) # PyQt6.QtGui.QColor(colors[ind])
            self._plots[plotInds[0]].plot(x,y, pen = (col.red(), col.green(), col.blue(), 255*0.33))
            if inte.foundPeak is not None and inte.foundPeak % 128:
                x = x[np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                y = y[np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                col = PyQt6.QtGui.QColor.fromRgb(self.__highlightColor1[0], self.__highlightColor1[1], self.__highlightColor1[2]) # PyQt6.QtGui.QColor(colors[ind])
                if inte.foundPeak is not None and inte.foundPeak % 128 == 2:
                    col = PyQt6.QtGui.QColor.fromRgb(self.__highlightColor2[0], self.__highlightColor2[1], self.__highlightColor2[2])
                self._plots[plotInds[0]].plot(x, y, pen = (col.red(), col.green(), col.blue(), 255))
        if refRT is not None:
            infLine = pyqtgraph.InfiniteLine(pos = [refRT, 0], movable=False, angle=90, label='', pen=self.__normalColor)
            self._plots[plotInds[0]].addItem(infLine)
            
        self._plots[plotInds[0]].setTitle(title)
        self._plots[plotInds[0]].setLabel('left', "Intensity")
        self._plots[plotInds[0]].setLabel('bottom', "Retention time (min)")
        
        for ind in range(len(intes)):
            inte = intes[ind]
            
            if inte.foundPeak is not None and inte.foundPeak % 128:
                try:
                    tempRT = inte.chromatogram["rts"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                    temp = inte.chromatogram["eic"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                    minVal = np.min(temp)
                    maxVal = np.max(temp - minVal)
                    peakApexRT = tempRT[np.argmax(temp)]

                    col = PyQt6.QtGui.QColor.fromRgb(self.__highlightColor1[0], self.__highlightColor1[1], self.__highlightColor1[2]) # PyQt6.QtGui.QColor(colors[ind])PyQt6.QtGui.QColor(colors[ind])
                    if inte.foundPeak is not None and inte.foundPeak % 128 == 2:
                        col = PyQt6.QtGui.QColor.fromRgb(self.__highlightColor2[0], self.__highlightColor2[1], self.__highlightColor2[2])
                    self._plots[plotInds[1]].plot(tempRT - peakApexRT, 
                                                (temp - minVal) / maxVal,
                                                pen = (col.red(), col.green(), col.blue(), 255*0.3))
                except Exception as ex:
                    logging.exception("Error in plotting integration")

        self._plots[plotInds[1]].setTitle(title + "; zoomed, aligned peaks")
        self._plots[plotInds[1]].setLabel('left', "Intensity")
        self._plots[plotInds[1]].setLabel('bottom', "Retention time (min)")
        
        self._plots[plotInds[0]].plotItem.scene().sigMouseClicked.connect(functools.partial(self.mouseClickedInPlot, plotItem = self._plots[plotInds[0]].plotItem))
        self._plots[plotInds[1]].plotItem.scene().sigMouseClicked.connect(functools.partial(self.mouseClickedInPlot, plotItem = self._plots[plotInds[0]].plotItem))
        
    def mouseClickedInPlot(self, evt, plotItem):
        pos = evt.scenePos()
        if plotItem.sceneBoundingRect().contains(pos):
            pos = plotItem.vb.mapSceneToView(pos)
            if self.lastExp is not None and self.lastSub is not None:
                closest = None
                closestDist = 1E6
                for samp, inte in self.loadedExperiments[self.lastExp].integrations[self.lastSub].items():
                    if not inte.other["GUIElement"].isHidden():
                        rts = inte.chromatogram["rts"]
                        eic = inte.chromatogram["eic"]
                        sigInd = np.argmin(np.abs(rts - pos.x()))
                        dist = np.abs(eic[sigInd] - pos.y())
                        if dist < closestDist:
                            closest = (samp, inte)
                            closestDist = dist
            self.tree.setCurrentItem(closest[1].other["GUIElement"])
            self.tree.scrollToItem(closest[1].other["GUIElement"])
        
    def plotCalibration(self, calInfo, title, unit = "NA", addLM = True, highlightLevel = None, plotInd = 6):
        try:
            self._plots[plotInd].plot([i for i, l in calInfo], [abs(l) for i, l in calInfo], pen=None, symbolSize=8, symbolBrush=(self.__normalColor[0], self.__normalColor[1], self.__normalColor[2], int(255*0.33)), symbolPen='w')
            self._plots[plotInd].plot([i for i, l in calInfo if l > 0], [abs(l) for i, l in calInfo if l > 0], pen=None, symbolSize=8, symbolBrush=self.__normalColor, symbolPen='w')
            if highlightLevel is not None:
                for ci, (i, l) in enumerate(calInfo):
                    if l == highlightLevel:
                        self._plots[plotInd].plot([calInfo[ci][0]], [abs(calInfo[ci][1])], pen=None, symbolSize=8, symbolBrush=(self.__highlightColor1[0], self.__highlightColor1[1], self.__highlightColor1[2], int(255 * (1 if l > 0 else 0.33))), symbolPen='w')
            ax = self._plots[plotInd].getAxis('left')
            ax.setTicks([] + [[(abs(l), "%.3f"%(abs(l))) for i, l in calInfo + [(0,0)]]])          
            
            if len(calInfo) > 1 and addLM:
                usedAreas = [a for a, c in calInfo if c > 0]
                usedConcs = [c for a, c in calInfo if c > 0]
                if len(usedAreas) >= 2:
                    model, r2, yhat, params, strRepr = PeakBotMRM.calibrationRegression(usedAreas, usedConcs,
                                                                                        type = self.calibrationMethod.currentText())
                    useCals = [a for a, c in calInfo if c > 0]
                    if len(useCals) > 1:
                        useCals = sorted(useCals)
                        a = np.linspace(useCals[0], useCals[1], self.__calibrationFunctionstep)
                        if len(useCals) > 2:
                            for i in range(1, len(useCals) - 1):
                                a = np.concatenate((a, np.linspace(useCals[i], useCals[i+1], self.__calibrationFunctionstep)))
                    
                    b = None
                    calcCons = None
                    if self.calibrationMethod.currentText() in ["linear", "linear, 1/expConc."]:
                        b = model(np.array((a)).reshape(-1,1))
                        calcCons = model(np.array((usedAreas)).reshape(-1,1))
                        
                    elif self.calibrationMethod.currentText() in ["quadratic", "quadratic, 1/expConc."]:
                        b = model(np.array((a)).reshape(-1,1))[:,0]
                        calcCons = model(np.array((usedAreas)).reshape(-1,1))[:,0]
                        
                    self._plots[plotInd].plot(a, b, pen=self.__normalColor)
                    self._plots[plotInd].plot(usedAreas, calcCons, pen=None, symbolSize=8, symbolBrush="Orange", symbolPen='w')
                    infLine = pyqtgraph.InfiniteLine(pos = 0, movable=False, angle=0, label='', pen=self.__normalColor)
                    self._plots[plotInd].addItem(infLine)  
                            
                    self._plots[plotInd].setTitle(title + "; R2 %.3f; %d points"%(r2, len(calInfo)))
                    self._plots[plotInd].setLabel('bottom', "Value")
                    self._plots[plotInd].setLabel('left', "Expected (%s)"%(unit))
        except:
            PyQt6.QtWidgets.QMessageBox.critical(self, "PeakBotMRM", "<b>Critical problem</b><br><br>A critical problem happend. Please save your work, take a screenshot, document what you have been doing, contact the developers, and then continue in a new Window. <br><br>We apololgize for this inconvenience!")
            logging.exception("Exception in plotCalibration")
            
    def closeEvent(self, event):
        close = PyQt6.QtWidgets.QMessageBox.question(self,
                                        "PeakBotMRM",
                                        "Are you sure want to quit?<br>Any unsaved changes will be lost",
                                        PyQt6.QtWidgets.QMessageBox.StandardButton.Yes | PyQt6.QtWidgets.QMessageBox.StandardButton.No)
        if close == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    
    


# To prevent random crashes when rerunning the code,
# first check if there is instance of the app before creating another.
app = PyQt6.QtWidgets.QApplication.instance()
if app is None:
    app = PyQt6.QtWidgets.QApplication(sys.argv)
    #from qt_material import apply_stylesheet
    #apply_stylesheet(app, theme='dark_red.xml', extra = { 'density_scale': '-4' })

main = Window()
main.showMaximized()

try:
    #main.loadExperiment("R100140", "./Reference/transitions.tsv", "./Reference/R100140_METAB02_MCC025_20200306", None, "\t")
    #main.loadExperiment("Ref_R100140", "./Reference/transitions.tsv", "./Reference/R100140_METAB02_MCC025_20200306", "./Reference/R100140_Integrations.csv", ",")
    #main.loadExperiment("R100138", "./Reference/transitions.tsv", "./Reference/R100138_METAB02_MCC025_20200304", None, ",")
    #main.loadExperiment("Ref_R100138", "./Reference/transitions.tsv", "./Reference/R100138_METAB02_MCC025_20200304", "./Reference/R100138_Integrations.csv", ",")
    
    if False:
        for expName, folder1, rawFolder in [("R100146", "validation", "R100146_METAB02_MCC025_20200403"), ("R100192", "validation", "R100192_METAB02_MCC025_20201125"), 
                                            ("R100210", "validation", "R100210_METAB02_MCC025_20210305"), ("R100147", "training", "R100147_METAB02_MCC025_20200409"), 
                                            ("R100194", "training", "R100194_METAB02_MCC025_20201203"), ("R100211", "training", "R100211_METAB02_MCC025_20210316"), 
                                            ("R100232", "training", "R100232_B_METAB02_MCC025_20210804")]:
            main.loadExperiment(expName, "./machine_learning_datasets_peakbot/%s/adaptedTransitions/%s.tsv"%(folder1, expName), "./machine_learning_datasets_peakbot/%s/%s"%(folder1, rawFolder), None, ",")
    
    if False:
        exps = [
            ("R100266", "R100266_METAB02_MCC025_20220218"), ("R100267", "R100267_METAB02_MCC025_20220225"), ("R100268", "R100268_METAB02_MCC025_20220304"),
            ("R100269", "R100269_METAB02_MCC025_20220311"), ("R100270", "R100270_METAB02_MCC025_20220325"), ("R100272", "R100272_METAB02_MCC025_20220401"),
            ("R100275", "R100275_METAB02_MCC025_20220421"), ("R100276", "R100276_METAB02_MCC025_20220427"), ("R100277", "R100277_METAB02_MCC025_20220505"),
            ("R100278", "R100278_METAB02_MCC025_20220512"), ("R100284", "R100284_METAB02_MCC025_20220519"), ("R100285", "R100285_METAB02_MCC025_20220610")
        ]
        
        for expName, rawFolder in exps:
            main.loadExperiment(expName, "./Reference/transitions.tsv", "./machine_learning_datasets_peakbot/unprocessed/%s"%(rawFolder), None, ",")
            
    if False:
        exps = [("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100131_METAB02_MCC025_20200225",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100138_METAB02_MCC025_20200304",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100142_METAB02_MCC025_20200317",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100147_METAB02_MCC025_20200409",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100150_METAB02_MCC0025_20200504", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100152_METAB02_MCC025_20200519",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100154_METAB02_MCC025_20200605",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100156_METAB02_MCC025_20200702",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100162_METAB02_MCC025_20200721",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100172_METAB02_MCC025_20200819",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100174_METAB02_MCC025_20200831",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100178_METAB02_MCC025_20200922",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100181_METAB02_MCC025_20201006",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100183_METAB02_MCC025_20201020",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100187_METAB02_MCC025_20201111",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100194_METAB02_MCC025_20201203",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100206_METAB02_MCC025_20210210",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100209_METAB02_MCC025_20210226",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100211_METAB02_MCC025_20210316",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100219_METAB02_MCC025_20210423",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100224_METAB02_MCC025_20210518",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100226_METAB02_MCC025_20210527",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100232_B_METAB02_MCC025_20210804","C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"),  
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/training/R100245_METAB02_MCC025_20210915",  "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"),  
                
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100134_METAB02_MCC025_20200228",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100140_METAB02_MCC025_20200306",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100146_METAB02_MCC025_20200403",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100151_METAB02_MCC025_20200507",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100153_METAB02_MCC025_20200612",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100155_METAB02_MCC025_20200623",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100159_MEATB02_MCC025_20200714",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100171_METAB02_MCC025_MCC059_20200814",    "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"),        
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100173_METAB02_MCC025_20200825",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100175_METAB02_MCC025_20200908",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100182_METAB02_MCC025_20201016",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100185_METAB02_MCC025_20201106",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100188_METAB02_MCC025_20201117",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100192_METAB02_MCC025_20201125",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100200_METAB02_MCC025_20201210",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100205_METAB02_MCC025_20210205",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100207_METAB02_MCC025_20210223",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100210_METAB02_MCC025_20210305",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100217_METAB02_MCC025_20210420",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100221_METAB02_MCC025_20210430",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100225_METAB02_MCC025_20210521",           "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/validation/R100236_METAB02_MCC025_LYSO_GABA_20210903", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"),           
                
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100246_METAB02_MCC025_20210924", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100248_METAB02_MCC025_20211007", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100249_METAB02_MCC025_20211012", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100251_METAB02_MCC025_20211021", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100252_METAB02_MCC025_20211028", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100254_METAB02_MCC025_20211105", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100256_METAB02_MCC025_20211203", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100258_METAB02_MCC025_20211210", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100260_METAB02_MCC025_20211217", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100261_METAB02_MCC025_20220131", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100262_METAB02_MCC025_20220209", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100238_METAB02_MCC025_20210730", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100239_METAB02_MCC025_20210806", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100240_METAB02_MCC025_20210819", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100245_METAB02_MCC025_20210915", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100266_METAB02_MCC025_20220218", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100267_METAB02_MCC025_20220225", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100268_METAB02_MCC025_20220304", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_UntilR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100269_METAB02_MCC025_20220311", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100270_METAB02_MCC025_20220325", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100272_METAB02_MCC025_20220401", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100275_METAB02_MCC025_20220421", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100276_METAB02_MCC025_20220427", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100277_METAB02_MCC025_20220505", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100278_METAB02_MCC025_20220512", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100284_METAB02_MCC025_20220519", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100285_METAB02_MCC025_20220610", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100286_METAB02_MCC025_20220615", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ("C:/Projects/PeakBot_MRM/PeakBotMRM_examples/machine_learning_datasets_peakbot/unprocessed/R100287_METAB02_MCC025_20220622", "C:/Projects/PeakBot_MRM/PeakBotMRM_examples/Reference/transitions_AfterInclR100269.tsv"), 
                ]
        for exp in exps:
            expName = os.path.basename(exp[0])
            expName = expName[ : expName.find("_")]
            main.loadExperiment(expName, 
                                exp[1], 
                                exp[0], None, ",")
        
    
    if False:
        main.loadExperiment("R100285", "./Reference/transitions.tsv", "./machine_learning_datasets_peakbot/unprocessed/R100285_METAB02_MCC025_20220610", None, ",")
        main.tree.setCurrentItem(main.tree.topLevelItem(0))
        main.processExperimentS(peakBotMRMModelFile = "C:/Users/cbueschl/AppData/Local/PeakBotMRM/models\METAB02__0a967796629c438387f2ba81482cd37e.h5", all = True, keepManualIntegrations = False)
    
    if True:
        binaryExps = [f for f in os.listdir(".") if os.path.isfile(f) and os.path.basename(f).lower().endswith(".pbexp")]
        if len(binaryExps) > 0:
            button = PyQt6.QtWidgets.QMessageBox.question(main, "Found experiments", "%d experiments have been found in the current working directory<br>'%s'"%(len(binaryExps), os.path.abspath(".").replace("\\", "/")))
            if button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                for binExp in binaryExps:
                    main.loadBinaryExperiment(binExp)
                main.tree.setCurrentItem(main.tree.topLevelItem(0))
            
    
except Exception as ex:
    print(ex)
    logging.exception("Exception in main window.")
    
    
app.exec()



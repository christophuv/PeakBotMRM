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
import PySimpleGUI as sg
try:
    splashImage = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui-resources", "robot_loading.png")
    window = sg.Window("", [[sg.Image(splashImage)]], transparent_color=sg.theme_background_color(), no_titlebar=True, keep_on_top=True, finalize=True)
    window.bring_to_front()
except:
    logging.warning("Cannot show splash screen")

import sys
from typing import OrderedDict
import functools

import os
import shutil
import sys
sys.path.append(os.path.join("..", "PeakBotMRM", "src"))
import natsort
import re
import pickle
import subprocess
from pathlib import Path

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
    for imp, typ in importantSamplesRegEx.items():
        for samp in natsort.natsorted(sampleNames):
            if re.search(imp, samp) and samp not in order:
                order.append(samp)
    
    for samp in natsort.natsorted(sampleNames):
        if samp not in order:
            order.append(samp)
    
    return order    


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
        

        
        
        




class TrainModelDialog(PyQt6.QtWidgets.QDialog):
    def __init__(self, parent=None, sampleOrder = None):
        super(TrainModelDialog, self).__init__(parent)
        
        if sampleOrder is None:
            sampleOrder = "{}"
        
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
        fDir = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder for model and log")
        if fDir:
            self.folderPath.setText(fDir)
    
    def getUserData(self):
        return (self.modelName.text(), self.traPath.text(), self.folderPath.text(), self.resPath.text())
    


class OpenExperimentDialog(PyQt6.QtWidgets.QDialog):
    def __init__(self, parent=None, sampleOrder = None):
        super(OpenExperimentDialog, self).__init__(parent)
        
        if sampleOrder is None:
            sampleOrder = "{}"
        
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
        grid.addWidget(PyQt6.QtWidgets.QLabel("Important samples"), rowi, 0)
        self.importantSamples = PyQt6.QtWidgets.QPlainTextEdit(str(sampleOrder))
        def cahnged():
            s = self.importantSamples.text()
            try:
                eval(s)
                self.importantSamples.setStyleSheet("border-color: Olivedrab; border-style: solid; border-width: 2px;")
            except:
                self.importantSamples.setStyleSheet("border-color: Firebrick; border-style: solid; border-width: 2px;")
        self.importantSamples.textChanged.connect(cahnged)
        grid.addWidget(self.importantSamples, rowi, 1, 2, 2)
        
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
        fName = PyQt6.QtWidgets.QFileDialog.getOpenFileName(self, "Open transitions file", filter="Tab separated values files (*.tsv);;Comma separated values files (*.csv);;All files (*.*)")
        if fName[0]:
            self.traPath.setText(fName[0])
        
    def openRaw(self):
        fDir = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder with raw LCMS data")
        if fDir:
            self.rawPath.setText(fDir)
    
    def openResults(self):
        fName = PyQt6.QtWidgets.QFileDialog.getOpenFileName(self, "Open results table", filter="Tab separated values files (*.tsv);;Comma separated values file (*.csv);;All files (*.*)")
        if fName[0]:
            self.resPath.setText(fName[0])
        
    def getUserData(self):
        return (self.expName.text(), self.traPath.text(), self.rawPath.text(), self.resPath.text(), self.delim.currentText().replace("TAB", "\t"), self.importantSamples.toPlainText())
    



        
                

class Window(PyQt6.QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        self._pyFilePath = os.path.dirname(os.path.abspath(__file__))
        
        self.__sampleNameReplacements = {"Ref_": "", "METAB02_": "", "MCC025_": "", "R100140_": "", "R100138_": ""}
        self.__leftPeakDefault = -0.1
        self.__rightPeakDefault = 0.1
        self.__importantSamplesRegEx = OrderedDict([("_CAL[0-9]+_", "CAL"), ("_NIST[0-9]+_", "NIST"), (".*", "sample")])
        self.__normalColor = (112,128,144)
        self.__highlightColor = (178,34,34)
        self.__msConvertPath = "msconvert" #"%LOCALAPPDATA%\\Apps\\ProteoWizard 3.0.22119.ba94f16 32-bit\\msconvert.exe"
        self.__calibrationFunctionstep = 100
        self.__exportSeparator = "\t"
        self.__defaultJumpWidth = 0.005
        
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
        grid.addWidget(self.infoLabel, 0, 0)
        self.hasPeak = PyQt6.QtWidgets.QCheckBox("Is peak")
        self.peakStart = PyQt6.QtWidgets.QDoubleSpinBox()
        self.peakStart.setDecimals(3); self.peakStart.setMaximum(100); self.peakStart.setMinimum(0); self.peakStart.setSingleStep(0.005)
        self.peakEnd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.peakEnd.setDecimals(3); self.peakEnd.setMaximum(100); self.peakEnd.setMinimum(0); self.peakEnd.setSingleStep(0.005)
        self.istdhasPeak = PyQt6.QtWidgets.QCheckBox("Is peak")
        self.istdpeakStart = PyQt6.QtWidgets.QDoubleSpinBox()
        self.istdpeakStart.setDecimals(3); self.istdpeakStart.setMaximum(100); self.istdpeakStart.setMinimum(0); self.istdpeakStart.setSingleStep(0.005)
        self.istdpeakEnd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.istdpeakEnd.setDecimals(3); self.istdpeakEnd.setMaximum(100); self.istdpeakEnd.setMinimum(0); self.istdpeakEnd.setSingleStep(0.005)
        self.useForCalibration = PyQt6.QtWidgets.QCheckBox()
        self.calibrationMethod = PyQt6.QtWidgets.QComboBox()
        self.calibrationMethod.addItems(["y=k*x+d", "y=k*x+d; 1/expConc.", "y=k*x**2+l*x+d", "y=k*x**2+l*x+d; 1/expConc."])
        self.calibrationMethod.setCurrentIndex(1)
        layout = PyQt6.QtWidgets.QHBoxLayout()
        
        layout.addWidget(PyQt6.QtWidgets.QLabel("Substance:"))
        layout.addWidget(self.hasPeak)
        self.hasPeak.stateChanged.connect(self.curFeatureChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("start (min)"))
        layout.addWidget(self.peakStart)
        self.peakStart.valueChanged.connect(self.curFeatureChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("end (min)"))
        layout.addWidget(self.peakEnd)
        self.peakEnd.valueChanged.connect(self.curFeatureChanged)
        layout.addStretch()
        
        layout.addWidget(PyQt6.QtWidgets.QLabel("ISTD:"))
        layout.addWidget(self.istdhasPeak)
        self.istdhasPeak.stateChanged.connect(self.curFeatureChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("start (min)"))
        layout.addWidget(self.istdpeakStart)
        self.istdpeakStart.valueChanged.connect(self.curFeatureChanged)
        layout.addWidget(PyQt6.QtWidgets.QLabel("end (min)"))
        layout.addWidget(self.istdpeakEnd)
        self.istdpeakEnd.valueChanged.connect(self.curFeatureChanged)
        layout.addStretch()
        
        layout.addWidget(PyQt6.QtWidgets.QLabel("Use for calibration"))
        layout.addWidget(self.useForCalibration)
        self.useForCalibration.stateChanged.connect(self.curFeatureChanged)
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
        self.tree = PyQt6.QtWidgets.QTreeWidget()
        self.tree.currentItemChanged.connect(self.treeClicked)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Generic", "Area", "PeakWidth"])
        self.tree.setMinimumWidth(300)
        self.tree.header().resizeSection(0, 250)
        self.tree.header().resizeSection(1, 50)
        self.tree.header().resizeSection(2, 50)
        dock = Dock("Experiments")
        self.docks.append(dock)
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

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "folder-open-outline.svg")), "New / open", self)
        item.triggered.connect(self.userSelectExperimentalData)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "close-circle-outline.svg")), "Close experiment", self)
        item.triggered.connect(self.closeExperiment)
        toolbar.addAction(item)
        
        toolbar.addSeparator()

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")), "Run PeakBotMRM detection on active experiment", self)
        item.triggered.connect(self.processActiveExperimentEventHelper)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "list-circle-outline.svg")), "Show summary of active experiment", self)
        item.triggered.connect(self.showSummary)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "download-outline.svg")), "Export active experiment results", self)
        item.triggered.connect(functools.partial(self.exportIntegrations, all = False))
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "thunderstorm-outline.svg")), "Reset instances", self)
        item.triggered.connect(self.resetActivateExperiment)
        toolbar.addAction(item)
        
        toolbar.addSeparator()

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robots.png")), "Run PeakBotMRM detection on all openend experiments", self)
        item.triggered.connect(self.processAllExperimentEventHelper)
        toolbar.addAction(item)

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "code-download-outline.svg")), "Export all experiment results", self)
        item.triggered.connect(functools.partial(self.exportIntegrations, all = True))
        toolbar.addAction(item)
        
        toolbar.addSeparator()
        
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
                
        toolbar.addSeparator()
        
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot_grey.png")), "Train new model with active experiment", self)
        item.triggered.connect(self.trainNewModel)
        toolbar.addAction(item)
                
        toolbar.addSeparator()
        
        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "settings-outline.svg")), "Settings", self)
        item.triggered.connect(self.showSettings)
        toolbar.addAction(item)
        
        toolbar.addSeparator()

        item = PyQt6.QtGui.QAction(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "information-circle-outline.svg")), "Information", self)
        item.triggered.connect(self.showPeakBotMRMInfo)
        toolbar.addAction(item)
        
        self._substances = {}
        self._integrations = {}
        self._sampleInfo = {}
        
        self.lastExp = ""
        self.lastSub = ""
        self.lastSam = ""
        
        self.paCMAP = None
        
        self.tree._keyPressEvent = self.tree.keyPressEvent
        self.tree.keyPressEvent = self.keyPressEvent
            
        if os.path.exists(os.path.join(self._pyFilePath, "defaultSettings.pickle")):
            self.tree.blockSignals(True)
            self.loadSettings(settingsFile = os.path.join(self._pyFilePath, "defaultSettings.pickle"))
            self.tree.blockSignals(False)

        try:
            subprocess.run(self.__msConvertPath, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as ex:
            logging.error("\033[91mError: msconvert (%s) not found.\033[0m Download and install from https://proteowizard.sourceforge.io/")
            PyQt6.QtWidgets.QMessageBox.critical(None, "PeakBotMRM", "Error<br><br>MSConvert (at '%s') cannot be found. Please verify that it is present/installed and/or set the path to the executible accordingly<br><br>Download MSconvert from <a href='https://proteowizard.sourceforge.io/'>https://proteowizard.sourceforge.io/</a>.<br>Choose the version that is 'able to convert ventdor files'.<br>Install the software.<br>Then try restarting PeakBotMRM. If 'msconvert' alone does not work, try '%%LOCALAPPDATA%%\\Apps\\ProteoWizard 3.0.22119.ba94f16 32-bit\\msconvert.exe'"%(self.__msConvertPath))
    
    def keyPressEvent(self, event):
        if event.key() in [PyQt6.QtCore.Qt.Key.Key_Down, PyQt6.QtCore.Qt.Key.Key_Up]:
            self.tree._keyPressEvent(event)
        
        elif chr(event.key()) in ["q", "Q"]:
            self.hasPeak.setChecked(not self.hasPeak.isChecked())
        elif chr(event.key()) in ["w", "W"]:
            self.peakStart.setValue(self.peakStart.value() - self.__defaultJumpWidth)
        elif chr(event.key()) in ["e", "E"]:
            self.peakStart.setValue(self.peakStart.value() + self.__defaultJumpWidth)
        elif chr(event.key()) in ["r", "R"]:
            self.peakEnd.setValue(self.peakEnd.value() - self.__defaultJumpWidth)
        elif chr(event.key()) in ["t", "T"]:
            self.peakEnd.setValue(self.peakEnd.value() + self.__defaultJumpWidth)
        
        elif chr(event.key()) in ["a", "A"]:
            self.istdhasPeak.setChecked(not self.istdhasPeak.isChecked())
        elif chr(event.key()) in ["s", "S"]:
            self.istdpeakStart.setValue(self.istdpeakStart.value() - self.__defaultJumpWidth)
        elif chr(event.key()) in ["d", "D"]:
            self.istdpeakStart.setValue(self.istdpeakStart.value() + self.__defaultJumpWidth)
        elif chr(event.key()) in ["f", "F"]:
            self.istdpeakEnd.setValue(self.istdpeakEnd.value() - self.__defaultJumpWidth)
        elif chr(event.key()) in ["g", "G"]:
            self.istdpeakEnd.setValue(self.istdpeakEnd.value() + self.__defaultJumpWidth)
            
        
        return 
        self.hasPeak = PyQt6.QtWidgets.QCheckBox("Is peak")
        self.peakStart = PyQt6.QtWidgets.QDoubleSpinBox()
        self.peakStart.setDecimals(3); self.peakStart.setMaximum(100); self.peakStart.setMinimum(0); self.peakStart.setSingleStep(0.005)
        self.peakEnd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.peakEnd.setDecimals(3); self.peakEnd.setMaximum(100); self.peakEnd.setMinimum(0); self.peakEnd.setSingleStep(0.005)
        self.istdhasPeak = PyQt6.QtWidgets.QCheckBox("Is peak")
        self.istdpeakStart = PyQt6.QtWidgets.QDoubleSpinBox()
        self.istdpeakStart.setDecimals(3); self.istdpeakStart.setMaximum(100); self.istdpeakStart.setMinimum(0); self.istdpeakStart.setSingleStep(0.005)
        self.istdpeakEnd = PyQt6.QtWidgets.QDoubleSpinBox()
        self.istdpeakEnd.setDecimals(3); self.istdpeakEnd.setMaximum(100); self.istdpeakEnd.setMinimum(0); self.istdpeakEnd.setSingleStep(0.005)
        self.useForCalibration = PyQt6.QtWidgets.QCheckBox()
        self.calibrationMethod = PyQt6.QtWidgets.QComboBox()
        

    def saveSettings(self, settingsFile = None):
        if settingsFile is None:
            settingsFile = os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "defaultSettings.pickle")
            
        with open(settingsFile, "wb") as fout:
            settings = {
                    "PeakBotMRM.Config.RTSLICES": PeakBotMRM.Config.RTSLICES,
                    "PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN": PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN,
                    "PeakBotMRM.Config.INTEGRATIONMETHOD": PeakBotMRM.Config.INTEGRATIONMETHOD,
                    "PeakBotMRM.Config.CALIBRATIONMETHOD": PeakBotMRM.Config.CALIBRATIONMETHOD,
                    "PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT": PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT, 
                    "PeakBotMRM.Config.MRMHEADER": PeakBotMRM.Config.MRMHEADER,
                    
                    "GUI/__sampleNameReplacements": self.__sampleNameReplacements,
                    "GUI/__leftPeakDefault": self.__leftPeakDefault,
                    "GUI/__rightPeakDefault": self.__rightPeakDefault,
                    "GUI/__importantSamplesRegEx": self.__importantSamplesRegEx,
                    "GUI/__normalColor": self.__normalColor,
                    "GUI/__highlightColor": self.__highlightColor,
                    "GUI/__calibrationFunctionstep": self.__calibrationFunctionstep, 
                    "GUI/__exportSeparator": self.__exportSeparator.replace("\t", "TAB"), 
                    "GUI/__sortOrder": self.sortOrder.currentIndex(),
                    "GUI/__defaultJumpWidth": self.self.__defaultJumpWidth, 
                    
                    "GUI/DockAreaState": self.dockArea.saveState(),
                }
            pickle.dump(settings, fout)

    def loadSettings(self, settingsFile = None):
        if settingsFile is None:
            settingsFile = os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "defaultSettings.pickle")
            
        with open(settingsFile, "rb") as fin:
            settings = pickle.load(fin)
                
            PeakBotMRM.Config.RTSLICES = settings["PeakBotMRM.Config.RTSLICES"]
            PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN = settings["PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN"]
            PeakBotMRM.Config.INTEGRATIONMETHOD = settings["PeakBotMRM.Config.INTEGRATIONMETHOD"]
            PeakBotMRM.Config.CALIBRATIONMETHOD = settings["PeakBotMRM.Config.CALIBRATIONMETHOD"]
            PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT = settings["PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT"]
            PeakBotMRM.Config.MRMHEADER = settings["PeakBotMRM.Config.MRMHEADER"]
            
            self.__sampleNameReplacements = settings["GUI/__sampleNameReplacements"]
            self.__leftPeakDefault = settings["GUI/__leftPeakDefault"]
            self.__rightPeakDefault = settings["GUI/__rightPeakDefault"]
            self.__importantSamplesRegEx = settings["GUI/__importantSamplesRegEx"]
            self.__normalColor = settings["GUI/__normalColor"]
            self.__highlightColor = settings["GUI/__highlightColor"]
            self.__calibrationFunctionstep = settings["GUI/__calibrationFunctionstep"]
            self.__exportSeparator = settings["GUI/__exportSeparator"].replace("TAB", "\t")
            self.sortOrder.setCurrentIndex(settings["GUI/__sortOrder"])
            self.__defaultJumpWidth = settings["GUI/__defaultJumpWidth"]
            
            #self.dockArea.restoreState(settings["GUI/DockAreaState"])
    
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
                {'name': 'Calibration method', 'type': 'list', 'value': PeakBotMRM.Config.CALIBRATIONMETHOD, 'values': ['1', '1/expConcentration']},
                {'name': 'Calibration extend borders', 'type': 'bool', 'value': PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT},
                {'name': 'Calibration plot step size', 'type': 'int', 'value': self.__calibrationFunctionstep, 'step': 1, 'limits': [10, 1000]},
                {'name': 'MRM header', 'type': 'str', 'value': PeakBotMRM.Config.MRMHEADER},
            ]},
            {'name': 'Sample names', 'type': 'group', 'children':[
                {'name': 'Replacements', 'type': 'str', 'value': str(self.__sampleNameReplacements)},
                {'name': 'Important samples RegEx', 'type': 'str', 'value': str(self.__importantSamplesRegEx)},
            ]},
            {'name': 'New chromatographic peak (relative to ref. RT)', 'type': 'group', 'children': [
                {'name': 'Default left width', 'type': 'float', 'value': self.__leftPeakDefault, 'limits': [-1., 0], 'step' : .005, 'suffix': 'min'},
                {'name': 'Default right width', 'type': 'float', 'value': self.__rightPeakDefault, 'limits': [0., 1.], 'step' : .005, 'suffix': 'min'},
            ]},
            {'name': 'Plot colors', 'type': 'group', 'children':[
                {'name': 'Normal color', 'type': 'color', 'value': PyQt6.QtGui.QColor.fromRgb(*self.__normalColor)},
                {'name': 'Highlight color', 'type': 'color', 'value': PyQt6.QtGui.QColor.fromRgb(*self.__highlightColor)}
            ]},
            {'name': 'Other', 'type': 'group', 'children':[
                {'name': 'MSConvert executible', 'type': 'str', 'value': self.__msConvertPath, 'tip': 'Download MSconvert from <a href="https://proteowizard.sourceforge.io/">https://proteowizard.sourceforge.io/</a>. Choose the version that is "able to convert ventdor files". Install the software. Then try restarting PeakBotMRM. If "msconvert" alone does not work, try "%LOCALAPPDATA%\\Apps\\ProteoWizard 3.0.22119.ba94f16 32-bit\\msconvert.exe"'},
                {'name': 'Export delimiter', 'type': 'list', 'value': self.__exportSeparator, 'values': ["TAB", ",", ";", "$"]},
                {'name': 'Sort order', 'type': 'list', 'value': self.sortOrder.currentIndex(), 'values': ["Sort: Sample groups/names", "Sort: Peak area"]},
                {'name': 'Default jump width', 'type': 'float', 'value': self.__defaultJumpWidth, 'limits': [0.001, 0.2], 'suffix': 'min'}
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
            if p.param("Sample names", "Important samples RegEx") is None or p.param("Sample names", "Important samples RegEx") == "":
                self.__importantSamplesRegEx = {}
            else:
                try:
                    self.__importantSamplesRegEx = eval(p.param("Sample names", "Important samples RegEx").value())
                except:
                    PyQt6.QtWidgets.QMessageBox.critical(None, "PeakBotMRM", "Error<br><br>The entered sample importances are not a valid python dictionary (e.g. {'_CAL[0-9]+_':'Cal'}). Please modify.")
                    self.__importantSamplesRegEx = ["Error, invalid python object"]
            self.__leftPeakDefault = p.param("New chromatographic peak (relative to ref. RT)", "Default left width").value()
            self.__rightPeakDefault = p.param("New chromatographic peak (relative to ref. RT)", "Default right width").value()
            self.__normalColor = p.param("Plot colors", "Normal color").value().getRgb()
            self.__highlightColor = p.param("Plot colors", "Highlight color").value().getRgb()
            self.__calibrationFunctionstep = p.param("PeakBotMRM Configuration", "Calibration plot step size").value()
            self.__exportSeparator = p.param("Other", "Export delimiter").value().replace("TAB", "\t")
            self.sortOrder.setCurrentIndex(p.param("Other", "Sort order").value())
            self.__defaultJumpWidth = p.param("Other", "Default jump width").value()
                        
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
            self.loadSettings()
            self.showSettings()
        loadDefault.clicked.connect(action)
        layout.addWidget(loadDefault, 1, 0, 1, 1)
        
        saveAsDefault = PyQt6.QtWidgets.QPushButton("Accept and save as default")
        def action():
            dialog.accept()
            self.saveSettings()
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
        PyQt6.QtWidgets.QMessageBox.information(self, "PeakBotMRM", "<b>PeakBotMRM</b><br><br>PeakBotMRM was developed at <a href='https://cemm.at/research/facilities/molecular-discovery-platform/metabolomics-facility'>CEMM</a> and at the <a href='https://chemnet.univie.ac.at/'>University of Vienna</a>.<br> For further information please contact the authors.<br>(c) 2020 - 2022<br><br><b>Commercial use is prohibited!</b><br><br>Figures and illustrations have been desigend using resources from <a href='https://flaticon.com'>https://flaticon.com</a>")
    
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
        for mf in natsort.natsorted(list(os.listdir(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "PeakBotMRM", "models")))):
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
    
    def processExperimentS(self, peakBotMRMModelFile = None, all = False):
                
        if peakBotMRMModelFile is None:
            peakBotMRMModelFile = PyQt6.QtWidgets.QFileDialog.getOpenFileName(self, "Open PeakBotMRM model file", filter="PeakBotMRM models (*.h5)", directory=os.path.join(self._pyFilePath, "models"))
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
            for c in range(self.tree.topLevelItemCount()):
                it = self.tree.topLevelItem(c)
                selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            
                if selExp is not None and selExp != "" and selExp in self._integrations.keys():
                    expToProcess.append(selExp)
                
        else:
            l = list(self.tree.selectedItems())
            
            if len(l) == 1 and "experiment" in l[0].__dict__.keys():
                it = l[0]
                while it.parent() is not None:
                    it = it.parent()
            
                selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            
                if selExp is not None and selExp != "" and selExp in self._integrations.keys():
                    expToProcess.append(selExp)
                    
        button = PyQt6.QtWidgets.QMessageBox.question(self, "Process experiment", "<b>Warning</b><br><br>This will process the selected experiment(s) <br>'%s'<br> with PeakBotMRM (model '%s').<br> Do you want to keep any manual integrations?"%("', '".join(expToProcess), peakBotMRMModelFile), buttons = PyQt6.QtWidgets.QMessageBox.StandardButton.Yes | PyQt6.QtWidgets.QMessageBox.StandardButton.No | PyQt6.QtWidgets.QMessageBox.StandardButton.Abort, defaultButton = PyQt6.QtWidgets.QMessageBox.StandardButton.Yes)
        if button in [PyQt6.QtWidgets.QMessageBox.StandardButton.Yes, PyQt6.QtWidgets.QMessageBox.StandardButton.No]:
            
            for selExp in expToProcess:
                
                procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Predicting experiment '%s' with PeakBotMRM<br>Model: '%s'<br><br>See console for further details"%(selExp, os.path.basename(peakBotMRMModelFile)), 
                                                            minimum=0, maximum=len(self._integrations[selExp]))
                procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
                procDiag.setWindowTitle("PeakBotMRM")
                procDiag.setModal(True)
                procDiag.show()
                logging.info("")
                logging.info("")
                logging.info("Processing dataset '%s'"%(selExp))
                PeakBotMRM.predict.predictDataset(peakBotMRMModelFile, self._substances[selExp], self._integrations[selExp], callBackFunction = procDiag.setValue, showConsoleProgress = False)
                
                procDiag.hide()
                
                for s in self._integrations[selExp].keys():
                    for h in self._integrations[selExp][s]:
                        if (button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes and self._integrations[selExp][s][h].type != "Manual integration") or button == PyQt6.QtWidgets.QMessageBox.StandardButton.No:
                            self._integrations[selExp][s][h].type = self._integrations[selExp][s][h].other["pred.type"]
                            self._integrations[selExp][s][h].comment = self._integrations[selExp][s][h].other["pred.comment"]
                            
                            self._integrations[selExp][s][h].foundPeak = self._integrations[selExp][s][h].other["pred.foundPeak"]
                            if self._integrations[selExp][s][h].foundPeak:
                                self._integrations[selExp][s][h].rtStart = self._integrations[selExp][s][h].other["pred.rtstart"]
                                self._integrations[selExp][s][h].rtEnd = self._integrations[selExp][s][h].other["pred.rtend"]
                                self._integrations[selExp][s][h].area = self._integrations[selExp][s][h].other["pred.areaPB"]
            
                for c in range(self.tree.topLevelItemCount()):
                    it = self.tree.topLevelItem(c)
                    temp = it.experiment if "experiment" in it.__dict__.keys() else None
                    if temp == selExp:
                        it.setBackground(0, PyQt6.QtGui.QColor.fromRgb(255,255,255))
                        for helperInd in range(it.childCount()):
                            helper = it.child(helperInd)
                            for subNodeInd in range(helper.childCount()):
                                subNode = helper.child(subNodeInd)
                                for sampNodeInd in range(subNode.childCount()):
                                    sampleItem = subNode.child(sampNodeInd)
                                    inte = self._integrations[selExp][sampleItem.substance][sampleItem.sample]
                                    sampleItem.setText(1, str(inte.area) if inte.foundPeak else "")
                                    sampleItem.setText(2, "%.2f - %.2f"%(inte.rtStart, inte.rtEnd) if inte.foundPeak else "")
            
            if len(expToProcess) == 1:
                if PyQt6.QtWidgets.QMessageBox.question(self, "Generate summary", "Do you want to generate a summary of the processed results and detected chromatographic peaks?") == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    processingInfo = ["PeakBotMRM model file: %s"%(peakBotMRMModelFile), PeakBotMRM.Config.getAsStringFancy().replace("\n", "<br>")]
                    self.showSummary(processingInfo = processingInfo)
            else:    
                PyQt6.QtWidgets.QMessageBox.information(self, "Processed experiment(s)", "Experiment(s) has/have been processed. ")
            
    def showSummary(self, processingInfo = None):
        top = """
<!DOCTYPE html>
<html>
  <head>
    <title>Overview of results</title>
    <style>
      /* CSS styles */
      p {
        font-family: Arial;
      }
      h1, h2, h3 {
        font-family: Arial;
        color: Slategrey;
        border: solid 1px   #bdc3c7; 
        border-radius: 3px;
        padding-left: 6px;
        box-shadow: 2px 6px 6px lightgrey;
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
        if len(l) == 1 and "experiment" in l[0].__dict__.keys():
            with tempfile.TemporaryDirectory() as tmpDir:
                it = l[0]
                while it.parent() is not None:
                    it = it.parent()
                
                selExp = it.experiment if "experiment" in it.__dict__.keys() else None
                
                procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Generating summary of '%s'"%(selExp))
                procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
                procDiag.setWindowTitle("PeakBotMRM")
                procDiag.setModal(True)
                procDiag.show()

                body.append("<h1>Results of %s</h1>"%(selExp))
                
                if processingInfo is not None:
                    body.append("<h2>Processing info</h2>")
                    body.append("<p><ul><li>%s</li></ul></p>"%("</li><br><li>".join(processingInfo).replace("  | ..", "")))
                
                body.append("<h2>Sample statistics</h2>")
                
                ints = self._integrations[selExp]
                
                dat = {"peak": [], "peakWidth": [], "rtDeviation": [], "area": [], "substance": [], "sample": []}
                for sub in ints.keys():
                    refRT = self._substances[selExp][sub].refRT
                    for samp in ints[sub].keys():
                        dat["substance"].append(sub)
                        dat["sample"].append(samp)
                        inte = ints[sub][samp]
                        dat["peak"].append(inte.foundPeak)
                        if inte.foundPeak:
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

                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                temp_norm = StandardScaler().fit_transform(temp.to_numpy())
                temp = pd.DataFrame(temp_norm, columns = temp.columns, index = temp.index)
                pca = PCA(n_components=2)
                pca.fit(temp)
                pcaScores = pca.fit_transform(temp)
                temp = pd.DataFrame({"X": pcaScores[:,0],
                                     "Y": pcaScores[:,1],
                                     "sample": temp.index.values})
                
                p = (p9.ggplot(data = temp, mapping = p9.aes(x="X", y="Y", label="sample"))
                    #+ p9.geom_point()
                    + p9.geom_text()
                    + p9.ggtitle("PCA") + p9.xlab("Principal component 1") + p9.ylab("Principal component 2"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                body.append("<p>Note: Peak areas are used, no calibration has been carried out prior to the PCA. Z-scaling is used. Blank, calibration, STD samples are also included. </p>")
                
                body.append("<h2>Peak statistics</h2>")
                a = np.sum(dat["peak"])
                b = dat.shape[0] - a
                df = pd.DataFrame({"type": ["Peak", "no Peak"], "value": [a, b]})
                p = (p9.ggplot(data = df, mapping = p9.aes(x="type", y="value"))
                     + p9.geom_bar(stat="identity") 
                     + p9.coord_flip()
                     + p9.ggtitle("Detected peaks"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                body.append("<p>There are <b>%d (%.1f%%)</b> peaks and <b>%d (%.1f%%)</b> noPeaks in the experiment</p>"%(a, a/(a+b)*100, b, b/(a+b)*100))
                
                ## Peak width
                body.append("<h3>Peak width</h3>")
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="peakWidth"))
                    + p9.geom_histogram(bins = 100)
                    + p9.ggtitle("Peak width (min)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
            
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="peakWidth"))
                    + p9.geom_histogram(bins = 100)
                    + p9.xlim([0, 2])
                    + p9.ggtitle("Peak width zoomed (min)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="peakWidth", y="substance"))
                    + p9.geom_jitter(alpha = 0.1, width = 0, height = 0.2)
                    + p9.xlim([0, 2])
                    + p9.scales.scale_y_discrete(limits=list(dat[dat["peak"] == 1].groupby(["substance"]).mean().sort_values(["peakWidth"], ascending = False).reset_index()["substance"]))
                    + p9.ggtitle("Peak width zoomed (min)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 32/250*len(ints), units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                                
                
                ## Peak area
                body.append("<h3>Peak areas</h3>")
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="area"))
                    + p9.geom_histogram(bins = 1000)
                    + p9.scales.scale_x_log10()
                    + p9.ggtitle("Peak areas"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="area", y = "substance"))
                    + p9.geom_point(alpha=0.2)
                    + p9.scales.scale_y_discrete(limits=list(dat[dat["peak"] == 1].groupby(["substance"]).mean().sort_values(["area"], ascending = False).reset_index()["substance"]))
                    + p9.scales.scale_x_log10()
                    + p9.ggtitle("Peak areas"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 32/250*len(ints), units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                
                ## Peak apex relative to reference rt
                body.append("<h3>Peak retention times</h3>")
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="rtDeviation"))
                    + p9.geom_histogram(bins = 100)
                    + p9.ggtitle("Peak deviation (min; apex - reference)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                p = (p9.ggplot(data = dat[dat["peak"] == 1], mapping = p9.aes(x="rtDeviation", y="substance"))
                    + p9.geom_jitter(alpha=0.2, width=0, height = 0.2)
                    + p9.scales.scale_y_discrete(limits=list(dat[dat["peak"] == 1].groupby(["substance"]).mean().sort_values(["rtDeviation"], ascending = False).reset_index()["substance"]))
                    + p9.ggtitle("Rt deviation (min; apex - reference)"))
                p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 32/250*len(ints), units = "in", limitsize = False)
                body.append("<img src='data:image/png;base64,%s'></img><br>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
            
                for sub in natsort.natsorted(list(ints.keys())):
                    temp = dat[dat["peak"] == 1]
                    temp = temp[temp["substance"] == sub]
                    a = temp.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.9])
                    body.append("<h3>%s</h3>"%(sub))
                    body.append("<pre>%s</pre>"%(str(a).replace("\n", "<br>")))
                    
                    if False:
                        p = (p9.ggplot(data = temp[temp["substance"] == sub], mapping = p9.aes(x="peakWidth"))
                            + p9.geom_histogram(bins = 100)
                            + p9.xlim([0, 2])
                            + p9.ggtitle("Peak width zoomed for '%s' (min)"%(sub)))
                        p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                        body.append("<img src='data:image/png;base64,%s'></img>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                        
                        p = (p9.ggplot(data = temp[temp["substance"] == sub], mapping = p9.aes(x="area"))
                            + p9.geom_histogram(bins = 300)
                            + p9.scales.scale_x_log10()
                            + p9.ggtitle("Peak areas for '%s'"%(sub)))
                        p9.ggsave(plot=p, filename=os.path.join(tmpDir, "tempFig.png"), dpi = 72, width = 12, height = 8, units = "in", limitsize = False)
                        body.append("<img src='data:image/png;base64,%s'></img>"%(str(base64.b64encode(open(os.path.join(tmpDir, "tempFig.png"), "rb").read()))[2:-1]))
                
                procDiag.hide()

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
            
            it = it.child(0)
            
            for subI in range(it.childCount()):
                subit = it.child(subI)
                selSub = subit.substance
                
                ints = []
                sampleNames = []
                for sampI in range(subit.childCount()):
                    samp = subit.child(sampI)
                    selSamp = samp.sample
                    ints.append(self._integrations[selExp][selSub][selSamp])
                    sampleNames.append(selSamp)
                
                while subit.childCount() > 0:
                    samp = subit.child(0)
                    subit.removeChild(samp)
                
                order = None
                if sortMode == 1:
                    order = [i[0] for i in sorted(enumerate(ints), key = lambda x: x[1].area if x[1].foundPeak else 0)]
                elif sortMode == 2:
                    order = [i[0] for i in sorted(enumerate(ints), key = lambda x: x[1].area if x[1].foundPeak else 0, reverse = True)]
                else:
                    ## TODO apply important samples list for sorting similar to loadExperiments
                    #order = sortSamples(integrations[substance].keys(), importantSamples if importantSamples is not None else {}) 
                    #[i[0] for i in sorted(enumerate(sampleNames), key = lambda x: x[1])]
                    importantSamples = OrderedDict([("_CAL[0-9]+_", "CAL"), ("_NIST[0-9]+_", "NIST"), (".*", "sample")])
                    temp = sortSamples(sampleNames, importantSamples if importantSamples is not None else {}) 
                    order = []
                    for sampleName in temp:
                        order.append(sampleNames.index(sampleName))
                
                for i in order:
                    subit.addChild(ints[i].other["GUIElement"])
                    
    
    def resetActivateExperiment(self):
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "experiment" in l[0].__dict__.keys():
            it = l[0]
            while it.parent() is not None:
                it = it.parent()
            
            selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            
            if selExp is not None and selExp != "" and selExp in self._integrations.keys():
                
                button = PyQt6.QtWidgets.QMessageBox.question(self, "Reset experiment", "<b>Warning</b><br><br>This will reset the selected experiment (%s). Al progress (automated and manual annotations) will be lost. This action cannot be undone. <br>Are you sure that you want to continue?"%(selExp))
                if button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    
                    it.setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor[0]), int(self.__highlightColor[1]), int(self.__highlightColor[2]), int(255 * 0.2)))
                    for s in self._integrations[selExp].keys():
                        for h in self._integrations[selExp][s]:
                            self._integrations[selExp][s][h].foundPeak = False
                            if self._integrations[selExp][s][h].foundPeak:
                                self._integrations[selExp][s][h].rtStart = -1
                                self._integrations[selExp][s][h].rtEnd = -1
                                self._integrations[selExp][s][h].area = -1
                    
                    for helperInd in range(it.childCount()):
                        helper = it.child(helperInd)
                        for subNodeInd in range(helper.childCount()):
                            subNode = helper.child(subNodeInd)
                            for sampNodeInd in range(subNode.childCount()):
                                sampleItem = subNode.child(sampNodeInd)
                                inte = self._integrations[selExp][sampleItem.substance][sampleItem.sample]
                                sampleItem.setText(1, "")
                                sampleItem.setText(2, "")
                                
                    PyQt6.QtWidgets.QMessageBox.information(self, "Reset experiment", "Experiment '%s' has been reset. "%(selExp))
    
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
            outputFolder = PyQt6.QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")

        else:
            ls = list(self.tree.selectedItems())
            if len(ls) == 0:
                PyQt6.QtWidgets.QMessageBox.critical(self, "PeakBotMRM", "No experiment has been selected. Please select one from the list and retry.")
                return
            selExp = ls[0].experiment if "experiment" in ls[0].__dict__.keys() else None
            
            fName = PyQt6.QtWidgets.QFileDialog.getSaveFileName(self, "Save results to file", directory = os.path.join(".", "%s_PB%s"%(selExp, preExt)), filter = ext)
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
            
            selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            
            if outputFolder is not None:
                outputFile = os.path.join(outputFolder, "%s_PB%s"%(selExp, preExt))
            
            if selExp is not None and selExp != "" and selExp in self._integrations.keys():
                substancesComments, samplesComments = PeakBotMRM.predict.calibrateIntegrations(self._substances[selExp], self._integrations[selExp])
                
                PeakBotMRM.predict.exportIntegrations(outputFile, 
                                                      self._substances[selExp], 
                                                      self._integrations[selExp], 
                                                      separator = self.__exportSeparator, 
                                                      substancesComments = substancesComments, 
                                                      samplesComments = samplesComments, 
                                                      sampleMetaData = self._sampleInfo[selExp], 
                                                      additionalCommentsForFile=[],
                                                      oneRowHeader4Results = False)
                if len(ls) == 1:
                    procDiag.hide()
                    PyQt6.QtWidgets.QMessageBox.information(self, "Exporting results", "Experiment '%s' has been exported to file<br>'%s'"%(selExp, outputFile))
        
        procDiag.hide()
        if len(ls) > 1:
            PyQt6.QtWidgets.QMessageBox.information(self, "Exporting results", "Experiment results have been exported")
                    
                
    def closeExperiment(self):
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "experiment" in l[0].__dict__.keys():
            it = l[0]
            while it.parent() is not None:
                it = it.parent()
            selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            
            if selExp is not None and selExp != "" and selExp in self._integrations.keys():
                button = PyQt6.QtWidgets.QMessageBox.question(self, "Close experiment", "<b>Warning</b><br><br>This will close the selected experiment (%s). Any unsaved changes will be lost. This action cannot be undone. <br>Are you sure that you want to continue?"%(selExp))
                
                if button == PyQt6.QtWidgets.QMessageBox.StandardButton.Yes:
                    self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))
                    
                    del self._substances[selExp]
                    del self._integrations[selExp]
                    del self._sampleInfo[selExp]
                    
                    PyQt6.QtWidgets.QMessageBox.information(self, "Closed experiment", "Experiment '%s' has been closed."%(selExp))
        
    def genPlot(self):
        plot = pyqtgraph.PlotWidget()
        self._plots.append(plot)
        return plot
    
    def userSelectExperimentalData(self):
        dialog = OpenExperimentDialog(parent = self, sampleOrder = self.__importantSamplesRegEx)
        dialog.setModal(True)
        okay = dialog.exec()
        if okay:
            expName, transitionsFile, rawDataPath, resultsFile, delimChar, importantSamples = dialog.getUserData()
            main.loadExperiment(expName, transitionsFile, rawDataPath, resultsFile, delimChar, importantSamples = eval(importantSamples))
                
    def loadExperiment(self, expName, transitionFile, rawDataPath, integrationsFile, delimChar, importantSamples = None):
        self.tree.blockSignals(True)
        
        procDiag = PyQt6.QtWidgets.QProgressDialog(self, labelText="Loading experiment '%s'"%(expName))
        procDiag.setWindowIcon(PyQt6.QtGui.QIcon(os.path.join(self._pyFilePath, "gui-resources", "robot.png")))
        procDiag.setWindowTitle("PeakBotMRM")
        procDiag.setModal(True)
        procDiag.show()
        
        substances = PeakBotMRM.loadTargets(transitionFile, 
                                            logPrefix = "  | ..")
        
        integrations = None
        integrationsLoaded = False
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
                                                                            errorCallback = functools.partial(PyQt6.QtWidgets.QMessageBox.critical, self, "PeakBotMRM - conversion to mzML"))
        self._substances[expName]   = substances
        self._integrations[expName] = integrations
        self._sampleInfo[expName]   = sampleInfo
        
        if importantSamples is not None:
            for k, v in self._sampleInfo[expName].items():
                typ = "other"
                for h, j in importantSamples.items():
                    if re.search(h, k):
                        typ = j
                        break
                self._sampleInfo[expName][k]["Type"] = j
        for k, v in self._sampleInfo[expName].items():
            self._sampleInfo[expName][k]["Name"] = Path(self._sampleInfo[expName][k]["path"]).stem
            self._sampleInfo[expName][k]["Data File"] = os.path.basename(self._sampleInfo[expName][k]["path"])
        
        rootItem = PyQt6.QtWidgets.QTreeWidgetItem(self.tree)
        rootItem.setText(0, expName)
        if not integrationsLoaded:
            rootItem.setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor[0]), int(self.__highlightColor[1]), int(self.__highlightColor[2]), int(255 * 0.2)))
        rootItem.experiment = expName; rootItem.substance = None; rootItem.sample = None
        
        if True:
            substancesItem = PyQt6.QtWidgets.QTreeWidgetItem(rootItem)
            substancesItem.setText(0, "Substances")
            substancesItem.experiment = expName; substancesItem.substance = None; substancesItem.sample = None
                        
            allSamples = []
            curi = 0
            for substance in natsort.natsorted(integrations.keys()):
                procDiag.setValue(curi)
                curi = curi + 1
                
                substanceItem = PyQt6.QtWidgets.QTreeWidgetItem(substancesItem)
                substanceItem.experiment = expName; substanceItem.substance = substance; substanceItem.sample = None; substanceItem.userType = "All samples"
                substanceItem.setText(0, substance)
                if substance in self._substances[expName].keys():
                    s = self._substances[expName][substance].internalStandard
                    if s is not None and s != "":
                        substanceItem.setText(1, s)
                else:
                    substanceItem.setText(1, "Not found in transition list")
                
                for sample in sortSamples(integrations[substance].keys(), importantSamples if importantSamples is not None else {}):
                    inte = integrations[substance][sample]
                    sampleItem = PyQt6.QtWidgets.QTreeWidgetItem(substanceItem)
                    sampleItem.experiment = expName; sampleItem.substance = substance; sampleItem.sample = sample; sampleItem.userType = "Single peak"
                    showName = sample
                    for temp, rep in self.__sampleNameReplacements.items():
                        showName = showName.replace(temp, rep)
                    sampleItem.setText(0, showName)
                    sampleItem.setText(1, str(inte.area) if inte.foundPeak else "")
                    sampleItem.setText(2, "%.2f - %.2f"%(inte.rtStart, inte.rtEnd) if inte.foundPeak else "")
                    inte.other["GUIElement"] = sampleItem
                    
                    allSamples.append(sample)
            
        procDiag.close()
        
        self.tree.blockSignals(False)
        
    def curInterpolationFunctionChanged(self):
        self.tree.blockSignals(True); self.hasPeak.blockSignals(True); self.peakStart.blockSignals(True); self.peakEnd.blockSignals(True); self.istdhasPeak.blockSignals(True); self.istdpeakStart.blockSignals(True); self.istdpeakEnd.blockSignals(True); self.useForCalibration.blockSignals(True); self.calibrationMethod.blockSignals(True);
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "userType" in l[0].__dict__.keys():
            it = l[0]
            selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            selSub = it.substance if "substance" in it.__dict__.keys() else None
            if selExp is not None and selSub is not None:
                self._substances[selExp][selSub].calibrationMethod = self.calibrationMethod.currentText()
                self.curFeatureChanged()
        
        self.tree.blockSignals(False); self.hasPeak.blockSignals(False); self.peakStart.blockSignals(False); self.peakEnd.blockSignals(False); self.istdhasPeak.blockSignals(False); self.istdpeakStart.blockSignals(False); self.istdpeakEnd.blockSignals(False); self.useForCalibration.blockSignals(False); self.calibrationMethod.blockSignals(False);
    
    def curFeatureChanged(self):
        self.tree.blockSignals(True); self.hasPeak.blockSignals(True); self.peakStart.blockSignals(True); self.peakEnd.blockSignals(True); self.istdhasPeak.blockSignals(True); self.istdpeakStart.blockSignals(True); self.istdpeakEnd.blockSignals(True); self.useForCalibration.blockSignals(True); self.calibrationMethod.blockSignals(True);
        l = list(self.tree.selectedItems())
        if len(l) == 1 and "userType" in l[0].__dict__.keys() and l[0].userType == "Single peak":
            it = l[0]
            selExp = it.experiment if "experiment" in it.__dict__.keys() else None
            selSub = it.substance if "substance" in it.__dict__.keys() else None
            selSam = it.sample if "sample" in it.__dict__.keys() else None
            selIST = self._substances[selExp][selSub].internalStandard if selExp is not None and selSub is not None and self._substances[selExp][selSub].internalStandard is not None and self._substances[selExp][selSub].internalStandard != "" else None
            
            if selExp is not None and selSub is not None:
                self.calibrationMethod.setCurrentIndex(self.calibrationMethod.findText(self._substances[selExp][selSub].calibrationMethod))
            
            ## TODO improve this selection and move it to a generic format with __importantSamplesRegEx
            if "_CAL" in selSam:
                m = re.search("(_CAL[0-9]+_)", selSam)
                x = m.group(0)
                if x in self._substances[selExp][selSub].calSamples.keys():
                    self._substances[selExp][selSub].calSamples[x] = abs(self._substances[selExp][selSub].calSamples[x]) * (1 if self.useForCalibration.isChecked() else -1)

            inte = self._integrations[selExp][selSub][selSam]
            peakSwitched = inte.foundPeak != self.hasPeak.isChecked()
            inte.foundPeak = self.hasPeak.isChecked()
            if inte.foundPeak:
                eic = inte.chromatogram["eic"]
                rts = inte.chromatogram["rts"]
                
                if self.peakStart.value() < 0.05:
                    self.peakStart.setValue(self._substances[selExp][selSub].refRT + self.__leftPeakDefault)
                if self.peakEnd.value() < 0.05:
                    self.peakEnd.setValue(self._substances[selExp][selSub].refRT + self.__rightPeakDefault)
                
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
                it.setText(1, str(inte.area))
            else:
                it.setText(1, "")
            inte.type = "Manual integration"
            inte.comment = ""
            inte.other["GUIElement"].setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor[0]), int(self.__highlightColor[1]), int(self.__highlightColor[2]), int(255 * 0.2)))
                
            if selIST is not None:
                inte = self._integrations[selExp][selIST][selSam]
                peakSwitched = inte.foundPeak != self.hasPeak.isChecked()
                inte.foundPeak = self.istdhasPeak.isChecked()
                if inte.foundPeak:
                    eic = inte.chromatogram["eic"]
                    rts = inte.chromatogram["rts"]
                    
                    if self.istdpeakStart.value() < 0.05:
                        self.istdpeakStart.setValue(self._substances[selExp][selIST].refRT - 0.25)
                    if self.istdpeakEnd.value() < 0.05:
                        self.istdpeakEnd.setValue(self._substances[selExp][selIST].refRT + 0.25)
                        
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
                    inte.other["GUIElement"].setText(1, str(inte.area))
                else:
                    inte.other["GUIElement"].setText(1, "")
                    ## TODO set area in tree for the ISTD
                    
                inte.type = "Manual integration"
                inte.comment = ""            
                inte.other["GUIElement"].setBackground(0, PyQt6.QtGui.QColor.fromRgb(int(self.__highlightColor[0]), int(self.__highlightColor[1]), int(self.__highlightColor[2]), int(255 * 0.2)))
               
        if len(l) == 1 and "userType" in l[0].__dict__.keys():
            self.treeClicked(l[0], 0)
        self.tree.blockSignals(False); self.hasPeak.blockSignals(False); self.peakStart.blockSignals(False); self.peakEnd.blockSignals(False); self.istdhasPeak.blockSignals(False); self.istdpeakStart.blockSignals(False); self.istdpeakEnd.blockSignals(False);  self.useForCalibration.blockSignals(False); self.calibrationMethod.blockSignals(False);
    
    def refreshViews(self):
        self.lastExp = None
        self.lastSam = None
        self.lastSub = None
        
        for plot in self._plots:
            plot.enableAutoRange()
        
        self.curFeatureChanged()
    
    def treeClicked(self, it, col):
        self.tree.blockSignals(True); self.hasPeak.blockSignals(True); self.peakStart.blockSignals(True); self.peakEnd.blockSignals(True); self.istdhasPeak.blockSignals(True); self.istdpeakStart.blockSignals(True); self.istdpeakEnd.blockSignals(True); self.useForCalibration.blockSignals(True); self.calibrationMethod.blockSignals(True);
        selExp = it.experiment if "experiment" in it.__dict__.keys() else None
        selSub = it.substance if "substance" in it.__dict__.keys() else None
        selSam = it.sample if "sample" in it.__dict__.keys() else None
        selIST = self._substances[selExp][selSub].internalStandard if selExp is not None and selSub is not None and self._substances[selExp][selSub].internalStandard is not None and self._substances[selExp][selSub].internalStandard != "" else None
        
        for i, plot in enumerate(self._plots):
            if i in [1,2,4,5,9] and selExp == self.lastExp and selSub == self.lastSub:
                pass
            else:
                plot.clear()
        
        self.hasPeak.setChecked(False); self.peakStart.setValue(0); self.peakEnd.setValue(0); self.istdhasPeak.setChecked(False); self.istdpeakStart.setValue(0); self.istdpeakEnd.setValue(0)
         
        self.infoLabel.setText("Selected: Experiment <b>%s</b>, Substance <b>%s</b> (ISTD <b>%s</b>), Sample <b>%s</b>"%(
            selExp if selExp is not None else "-",
            selSub if selSub is not None else "-",
            selIST if selIST is not None else "-",
            selSam if selSam is not None else "-"
        ))
        
        if "userType" in it.__dict__:
            if it.userType == "Single peak":
        
                inte = self._integrations[selExp][selSub][selSam]
                self.infoLabel.setText("%s<br>Sub integration <b>%s%s</b>"%(self.infoLabel.text(), inte.type, " (%s)"%(inte.comment) if inte.comment is not None and inte.comment != "" else ""))
                for sampPart, level in self._substances[selExp][selSub].calSamples.items():
                    if sampPart in selSam:
                        self.useForCalibration.setChecked(level > 0)
                
                if inte.foundPeak:
                    self.hasPeak.setChecked(True)
                    self.peakStart.setValue(inte.rtStart)
                    self.peakEnd.setValue(inte.rtEnd)
                else: 
                    self.hasPeak.setChecked(False)
                self.plotIntegration(self._integrations[selExp][selSub][selSam], "Sub", refRT = self._substances[selExp][selSub].refRT, plotInd = 0)
                
                if selIST is not None :
                    inte = self._integrations[selExp][selIST][selSam]
                    self.infoLabel.setText("%s<br>ISTD integration <b>%s%s</b>"%(self.infoLabel.text(), inte.type, " (%s)"%(inte.comment) if inte.comment is not None and inte.comment != "" else ""))
                    if inte.foundPeak:
                        self.istdhasPeak.setChecked(True)
                        self.istdpeakStart.setValue(inte.rtStart)
                        self.istdpeakEnd.setValue(inte.rtEnd)
                    else:
                        self.istdhasPeak.setChecked(False)
                    self.plotIntegration(inte, "ISTD", refRT = self._substances[selExp][selIST].refRT if selIST is not None and selIST in self._substances[selExp].keys() else None, plotInd = 3)
                it = it.parent()
        
        if selExp != self.lastExp or selSub != self.lastSub:
            if "userType" in it.__dict__:
                if it.userType == "All samples":
                    ints = []
                    intsIS = []
                    for oitInd in range(it.childCount()):
                        oit = it.child(oitInd)
                        if "userType" in oit.__dict__ and oit.userType == "Single peak":
                            ints.append(self._integrations[oit.experiment][oit.substance][oit.sample])
                        if oit.experiment in self._substances.keys() and oit.substance in self._substances[oit.experiment].keys() and self._substances[oit.experiment][oit.substance].internalStandard in self._integrations[oit.experiment].keys() and oit.sample in self._integrations[oit.experiment][self._substances[oit.experiment][oit.substance].internalStandard].keys():
                            intsIS.append(self._integrations[oit.experiment][self._substances[oit.experiment][oit.substance].internalStandard][oit.sample])
                    if len(ints) > 0:
                        self.plotIntegrations(ints, "All EICs Sub", refRT = self._substances[selExp][selSub].refRT, plotInds = [1,2])
                    if len(intsIS) > 0:
                        self.plotIntegrations(intsIS, "All EICs ISTD", refRT = self._substances[selExp][selIST].refRT if selIST is not None and selIST in self._substances[selExp].keys() else None, plotInds = [4,5])
                
        if "userType" in it.__dict__:
            if it.userType == "All samples" and it.substance is not None and it.substance in self._substances[it.experiment].keys():
                substance = self._substances[selExp][selSub]
                istd = self._substances[selExp][substance.internalStandard] if substance.internalStandard is not None and substance.internalStandard != "" and substance.internalStandard in self._substances[selExp].keys() else None
                
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
                            if self._integrations[oit.experiment][oit.substance][oit.sample].foundPeak:
                                subArea.append((self._integrations[oit.experiment][oit.substance][oit.sample].area, level * substance.calLevel1Concentration))
                                if oit.sample == selSam:
                                    highlightLevel = level * substance.calLevel1Concentration
                            if istd is not None:
                                if self._integrations[oit.experiment][istd.name][oit.sample].foundPeak:
                                    isArea.append((self._integrations[oit.experiment][istd.name][oit.sample].area, level * istd.calLevel1Concentration))
                                    if oit.sample == selSam:
                                        highlightLevelIS = level
                                    if self._integrations[oit.experiment][oit.substance][oit.sample].foundPeak and self._integrations[oit.experiment][istd.name][oit.sample].foundPeak: 
                                        subRatio.append((self._integrations[oit.experiment][oit.substance][oit.sample].area / self._integrations[oit.experiment][istd.name][oit.sample].area, level * substance.calLevel1Concentration))
                                        if oit.sample == selSam:
                                            highlightLevelRatio = level * substance.calLevel1Concentration
                self.calibrationMethod.setCurrentIndex(self.calibrationMethod.findText(self._substances[selExp][selSub].calibrationMethod))
                
                if len(subArea) > 0:
                    self.plotCalibration(subArea, "Sub; areas", addLM = True, highlightLevel = highlightLevel, plotInd = 6)
                if len(isArea) > 0:
                    self.plotCalibration(isArea, "ISTD, areas", addLM = False, highlightLevel = highlightLevelIS, plotInd = 7)
                if len(subRatio) > 0:
                    self.plotCalibration(subRatio, "Sub/ISTD", addLM = True, highlightLevel = highlightLevelRatio, plotInd = 8)
           
        
        if selExp != self.lastExp or self.paCMAP is None:
            self.paCMAPAllEICsS = []
            self.paCMAPAllRTsS = []
            self.paCMAPsubstances = []
            self.paCMAPSamples = []
            self.paCMAP = None
            self.polyROI = None
            for sub in self._integrations[selExp].keys():
                for samp in self._integrations[selExp][sub].keys():
                    inte = self._integrations[selExp][sub][samp]
                    if "eicStd" not in inte.chromatogram.keys():
                        rtsS, eicS = PeakBotMRM.extractStandardizedEIC(inte.chromatogram["eic"], inte.chromatogram["rts"], self._substances[selExp][sub].refRT)
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
            
            self.polyROI = pyqtgraph.PolyLineROI(poly, closed=True, pen=self.__highlightColor)
            self.polyROI.sigRegionChangeFinished.connect(self.updatePACMAPROI)
            self._plots[9].addItem(self.polyROI)

            mask = points_in_polygon(poly, points)
            self._plots[9].plot(self.paCMAP[mask, 0], self.paCMAP[mask, 1], pen=None, symbol='o', symbolPen="Orange", symbolSize=8, symbolBrush="Orange")
            
            ints = []
            for xi, m in enumerate(mask):            
                if m:
                    xsub = self.paCMAPsubstances[xi]
                    xsam = self.paCMAPSamples[xi]
                    ints.append(self._integrations[self.lastExp][xsub][xsam])
                    
            if len(ints) > 0:
                self._plots[1].clear()
                self._plots[2].clear()
                self.plotIntegrations(ints, "All EICs Sub", plotInds = [1,2], makeUniformRT = True, scaleEIC = True)
        elif addROI:
            self.polyROI = pyqtgraph.PolyLineROI([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]], closed=True, pen=self.__highlightColor)
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
            self._plots[9].plot(self.paCMAP[highlight, 0], self.paCMAP[highlight, 1], pen=None, symbol='o', symbolPen=(0, self.__highlightColor[1], self.__highlightColor[2], int(255*1)), symbolSize=4, symbolBrush=(0, self.__highlightColor[1], self.__highlightColor[2], int(255*1)))
        if highlightSingle is not None:
            self._plots[9].plot(self.paCMAP[[highlightSingle], 0], self.paCMAP[[highlightSingle], 1], pen=None, symbol='o', symbolPen=(self.__highlightColor[0], self.__highlightColor[1], self.__highlightColor[2], int(255*1)), symbolSize=4, symbolBrush=(self.__highlightColor[0], self.__highlightColor[1], self.__highlightColor[2], int(255*1)))
            
    def updatePACMAPROI(self):
        self.plotPaCMAP(self.lastSub, self.lastSam)
        
    def addPolyROItoPaCMAP(self):
        self.polyROI = None
        self.plotPaCMAP(self.lastSub, self.lastSam, addROI = True)
    
    def plotIntegration(self, inte, title, refRT = None, plotInd = 0):
        self._plots[plotInd].plot(inte.chromatogram["rts"], inte.chromatogram["eic"], pen = self.__normalColor)
        
        if inte.foundPeak:
            try:
                rts = inte.chromatogram["rts"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                eic = inte.chromatogram["eic"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                p = self._plots[plotInd].plot(rts, eic, pen = self.__highlightColor)
                a = np.min(eic)
                p1 = self._plots[plotInd].plot(rts, np.ones(rts.shape[0]) * a)
                brush = (self.__highlightColor[0], self.__highlightColor[1], self.__highlightColor[2], 255*0.2)
                fill = pyqtgraph.FillBetweenItem(p, p1, brush)
                self._plots[plotInd].addItem(fill)
            except:
                pass
            

        if refRT is not None:
            infLine = pyqtgraph.InfiniteLine(pos = [refRT, 0], movable=False, angle=90, label='', pen=self.__normalColor)
            self._plots[plotInd].addItem(infLine)
            
            
        self._plots[plotInd].setTitle(title)
        self._plots[plotInd].setLabel('left', "Intensity")
        self._plots[plotInd].setLabel('bottom', "Retention time (min)")
        
    def plotIntegrations(self, intes, title, refRT = None, plotInds = [1,2], makeUniformRT = False, scaleEIC = False):
        
        for ind in range(len(intes)):
            inte = intes[ind]
            x = inte.chromatogram["rts"]
            y = inte.chromatogram["eic"]
            if makeUniformRT:
                x = np.linspace(0, 1, x.shape[0])
            if scaleEIC:
                y = y - np.min(y)
                y = y / np.max(y)
            self._plots[plotInds[0]].plot(x,y, pen = self.__normalColor)
            if inte.foundPeak:
                x = x[np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                y = y[np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                self._plots[plotInds[0]].plot(x, y, pen = self.__highlightColor)
        if refRT is not None:
            infLine = pyqtgraph.InfiniteLine(pos = [refRT, 0], movable=False, angle=90, label='', pen=self.__normalColor)
            self._plots[plotInds[0]].addItem(infLine)
            
        self._plots[plotInds[0]].setTitle(title)
        self._plots[plotInds[0]].setLabel('left', "Intensity")
        self._plots[plotInds[0]].setLabel('bottom', "Retention time (min)")
        
        for ind in range(len(intes)):
            inte = intes[ind]
            
            if inte.foundPeak:
                tempRT = inte.chromatogram["rts"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                temp = inte.chromatogram["eic"][np.logical_and(inte.rtStart <= inte.chromatogram["rts"], inte.chromatogram["rts"] <= inte.rtEnd)]
                minVal = np.min(temp)
                maxVal = np.max(temp - minVal)
                peakApexRT = tempRT[np.argmax(temp)]

                self._plots[plotInds[1]].plot(tempRT - peakApexRT, 
                                              (temp - minVal) / maxVal,
                                              pen = self.__highlightColor)

        self._plots[plotInds[1]].setTitle(title + "; zoomed, aligned peaks")
        self._plots[plotInds[1]].setLabel('left', "Intensity")
        self._plots[plotInds[1]].setLabel('bottom', "Retention time (min)")
        
        ## TODO finish
        #self._plots[plotInds[0]].plotItem.scene().sigMouseMoved.connect(functools.partial(self.mouseMoved, plotItem = self._plots[plotInds[0]].plotItem))
        
    def mouseMoved(self, evt, plotItem):
        pos = evt
        if plotItem.sceneBoundingRect().contains(pos):
            mousePoint = plotItem.vb.mapSceneToView(pos)
            print(mousePoint.x(), mousePoint.y())
            ## TODO finish that here
        
    def plotCalibration(self, calInfo, title, addLM = True, highlightLevel = None, plotInd = 6):
        self._plots[plotInd].plot([abs(l) for i, l in calInfo], [i for i, l in calInfo], pen=None, symbolSize=8, symbolBrush=(self.__normalColor[0], self.__normalColor[1], self.__normalColor[2], int(255*0.33)), symbolPen='w')
        self._plots[plotInd].plot([abs(l) for i, l in calInfo if l > 0], [i for i, l in calInfo if l > 0], pen=None, symbolSize=8, symbolBrush=self.__normalColor, symbolPen='w')
        if highlightLevel is not None:
            for ci, (i, l) in enumerate(calInfo):
                if l == highlightLevel:
                    self._plots[plotInd].plot([abs(calInfo[ci][1])], [calInfo[ci][0]], pen=None, symbolSize=8, symbolBrush=(self.__highlightColor[0], self.__highlightColor[1], self.__highlightColor[2], int(255 * (1 if l > 0 else 0.33))), symbolPen='w')
        ax = self._plots[plotInd].getAxis('bottom')
        ax.setTicks([[(abs(l), str(abs(l))) for i, l in calInfo]])
        
        if len(calInfo) > 1 and addLM:
            usedAreas = [a for a, c in calInfo if c > 0]
            usedConcs = [c for a, c in calInfo if c > 0]
            model, r2, yhat, params, strRepr = PeakBotMRM.calibrationRegression(usedAreas, usedConcs,
                                                                                type = self.calibrationMethod.currentText())
            if self.calibrationMethod.currentText() in ["y=k*x+d", "y=k*x+d; 1/expConc."]:
                intercept, coef = params
                useCals = [a for a, c in calInfo if c > 0]
                if len(useCals) > 1:
                    useCals = sorted(useCals)
                    a = np.linspace(useCals[0], useCals[1], self.__calibrationFunctionstep)
                    if len(useCals) > 2:
                        for i in range(1, len(useCals) - 1):
                            a = np.concatenate((a, np.linspace(useCals[i], useCals[i+1], self.__calibrationFunctionstep)))
                    self._plots[plotInd].plot(model(np.array((a)).reshape(-1,1)), a, pen=self.__normalColor)
                    calcCons = model(np.array((usedAreas)).reshape(-1,1))
                    self._plots[plotInd].plot(calcCons, usedAreas, pen=None, symbolSize=8, symbolBrush="Orange", symbolPen='w')
                
            elif self.calibrationMethod.currentText() in ["y=k*x**2+l*x+d", "y=k*x**2+l*x+d; 1/expConc."]:
                coef2, coef1, intercept = params
                useCals = [a for a, c in calInfo if c > 0]
                if len(useCals) > 1:
                    useCals = sorted(useCals)
                    a = np.linspace(useCals[0], useCals[1], self.__calibrationFunctionstep)
                    if len(useCals) > 2:
                        for i in range(1, len(useCals) - 1):
                            a = np.concatenate((a, np.linspace(useCals[i], useCals[i+1], self.__calibrationFunctionstep)))
                    self._plots[plotInd].plot(model(np.array((a)).reshape(-1,1))[:,0], a, pen=self.__normalColor)
                    calcCons = model(np.array((usedAreas)).reshape(-1,1))
                    self._plots[plotInd].plot(calcCons[:,0], usedAreas, pen=None, symbolSize=8, symbolBrush="Orange", symbolPen='w')
            self._plots[plotInd].setTitle(title + "; R2 %.3f; %d points"%(r2, len(calInfo)))
            self._plots[plotInd].setLabel('left', "Value")
            self._plots[plotInd].setLabel('bottom', "Exp. concentration")
            
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
    #apply_stylesheet(app, theme='light_teal.xml', extra = { 'density_scale': '-4' })

main = Window()
main.showMaximized()

try:
    #main.loadExperiment("R100140", "./Reference/transitions.tsv", "./Reference/R100140_METAB02_MCC025_20200306", None, "\t", importantSamples = OrderedDict([("_CAL[0-9]+_", "CAL"), ("_NIST[0-9]+_", "NIST"), (".*", "sample")]))
    #main.loadExperiment("Ref_R100140", "./Reference/transitions.tsv", "./Reference/R100140_METAB02_MCC025_20200306", "./Reference/R100140_Integrations.csv", ",", importantSamples = OrderedDict([("_CAL[0-9]+_", "CAL"), ("_NIST[0-9]+_", "NIST"), (".*", "sample")]))
    #main.loadExperiment("R100138", "./Reference/transitions.tsv", "./Reference/R100138_METAB02_MCC025_20200304", None, ",")
    #main.loadExperiment("Ref_R100138", "./Reference/transitions.tsv", "./Reference/R100138_METAB02_MCC025_20200304", "./Reference/R100138_Integrations.csv", ",")
    
    if False:
        for expName, folder1, rawFolder in [("R100146", "validation", "R100146_METAB02_MCC025_20200403"), ("R100192", "validation", "R100192_METAB02_MCC025_20201125"), 
                                            ("R100210", "validation", "R100210_METAB02_MCC025_20210305"), ("R100147", "training", "R100147_METAB02_MCC025_20200409"), 
                                            ("R100194", "training", "R100194_METAB02_MCC025_20201203"), ("R100211", "training", "R100211_METAB02_MCC025_20210316"), 
                                            ("R100232", "training", "R100232_B_METAB02_MCC025_20210804")]:
            main.loadExperiment(expName, "./machine_learning_datasets_peakbot/%s/adaptedTransitions/%s.tsv"%(folder1, expName), "./machine_learning_datasets_peakbot/%s/%s"%(folder1, rawFolder), None, ",", importantSamples = OrderedDict([("_CAL[0-9]+_", "CAL"), ("_NIST[0-9]+_", "NIST"), (".*", "sample")]))
    
    if False:
        exps = [
            ("R100266", "R100266_METAB02_MCC025_20220218"), ("R100267", "R100267_METAB02_MCC025_20220225"), ("R100268", "R100268_METAB02_MCC025_20220304"),
            ("R100269", "R100269_METAB02_MCC025_20220311"), ("R100270", "R100270_METAB02_MCC025_20220325"), ("R100272", "R100272_METAB02_MCC025_20220401"),
            ("R100275", "R100275_METAB02_MCC025_20220421"), ("R100276", "R100276_METAB02_MCC025_20220427"), ("R100277", "R100277_METAB02_MCC025_20220505"),
            ("R100278", "R100278_METAB02_MCC025_20220512"), ("R100284", "R100284_METAB02_MCC025_20220519"), ("R100285", "R100285_METAB02_MCC025_20220610")
        ]
        
        for expName, rawFolder in exps:
            main.loadExperiment(expName, "./Reference/transitions.tsv", "./machine_learning_datasets_peakbot/unprocessed/%s"%(rawFolder), None, ",", 
                                importantSamples = OrderedDict([("_CAL[0-9]+_", "CAL"), ("_NIST[0-9]+_", "NIST"), (".*", "sample")]))
            
    
except:
    logging.exception()
app.exec()



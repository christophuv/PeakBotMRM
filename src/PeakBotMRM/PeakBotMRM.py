import logging
import warnings

from .core import *

import sys
import os
import pickle
import uuid
import re
import math
import random
import natsort
import traceback
from pathlib import Path
import subprocess

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import tqdm
import pymzml

from .AdditionalValidationSets import *
from .CustomLossesMetrices import *



#####################################
### Configuration class
##
class Config(object):
    """Base configuration class"""

    NAME    = "PeakBotMRM"
    VERSION = "0.9.20"

    RTSLICES       = 255   ## should be of 2^n-1
    NUMCLASSES     =   2   ## [Peak, noPeak]

    BATCHSIZE      =  32
    STEPSPEREPOCH  =   8
    EPOCHS         = 300

    DROPOUT        = 0.2
    UNETLAYERSIZES = [32,64,128,256]

    LEARNINGRATESTART              = 0.00015
    LEARNINGRATEDECREASEAFTERSTEPS = 5
    LEARNINGRATEMULTIPLIER         = 0.002
    LEARNINGRATEMINVALUE           = 3e-17

    INSTANCEPREFIX = "___PBsample_"
    
    UPDATEPEAKBORDERSTOMIN = True
    INTEGRATIONMETHOD = "minbetweenborders"
    INCLUDEMETAINFORMATION = False
    CALIBRATIONMETHOD = "y=k*x+d; 1/expConc."
    
    MRMHEADER = "- SRM SIC Q1=(\d+\.?\d*[eE]?-?\d+) Q3=(\d+\.?\d*[eE]?-?\d+) start=(\d+\.?\d*[eE]?-?\d+) end=(\d+\.?\d*[eE]?-?\d+)"

    @staticmethod
    def getAsStringFancy():
        return "\n  | ..".join([
            "  | .. %s"%(Config.NAME),
            " Version " + Config.VERSION,
            " Python: %s"%(sys.version),
            " Tensorflow: %s"%(tf.__version__),
            " Tensorflow Addons: %s"%(tfa.__version__),
            " Size of EIC: %d (scans)"%(Config.RTSLICES),
            " Number of peak-classes: %d"%(Config.NUMCLASSES),
            " Batchsize: %d, Epochs %d, StepsPerEpoch: %d"%(Config.BATCHSIZE, Config.EPOCHS, Config.STEPSPEREPOCH),
            " DropOutRate: %g"%(Config.DROPOUT),
            " UNetLayerSizes: %s"%(Config.UNETLAYERSIZES),
            " LearningRate: Start: %g, DecreaseAfter: %d steps, Multiplier: %g, min. rate: %g"%(Config.LEARNINGRATESTART, Config.LEARNINGRATEDECREASEAFTERSTEPS, Config.LEARNINGRATEMULTIPLIER, Config.LEARNINGRATEMINVALUE),
            " Prefix for instances: '%s'"%Config.INSTANCEPREFIX,
            " Update peak borders to min: '%s'"%Config.UPDATEPEAKBORDERSTOMIN,
            " Integration method: '%s'"%Config.INTEGRATIONMETHOD,
            " Calibration method: '%s'"%Config.CALIBRATIONMETHOD,
        ])

    @staticmethod
    def getAsString():
        return ";".join([
            "%s"%(Config.NAME),
            "Version " + Config.VERSION,
            "Python: %s"%(sys.version),
            "Tensorflow: %s"%(tf.__version__),
            "Size of EIC: %d (scans)"%(Config.RTSLICES),
            "Number of peak-classes: %d"%(Config.NUMCLASSES),
            "Batchsize: %d, Epochs %d, StepsPerEpoch: %d"%(Config.BATCHSIZE, Config.EPOCHS, Config.STEPSPEREPOCH),
            "DropOutRate: %g"%(Config.DROPOUT),
            "UNetLayerSizes: %s"%(Config.UNETLAYERSIZES),
            "LearningRate: Start: %g, DecreaseAfter: %d steps, Multiplier: %g, min. rate: %g"%(Config.LEARNINGRATESTART, Config.LEARNINGRATEDECREASEAFTERSTEPS, Config.LEARNINGRATEMULTIPLIER, Config.LEARNINGRATEMINVALUE),
            "InstancePrefix: '%s'"%(Config.INSTANCEPREFIX),
            "Update peak borders to min: '%s'"%Config.UPDATEPEAKBORDERSTOMIN,
            "Integration method: '%s'"%Config.INTEGRATIONMETHOD,
            "Calibration method: '%s'"%Config.CALIBRATIONMETHOD,
        ])


def getTensorflowVersion():
    return tf.__version__
def getCPUInfo():
    try:
        import cpuinfo
        s = cpuinfo.get_cpu_info()["brand_raw"]
        return s
    except Exception:
        return "NA"
def getCUDAInfo():
    try:
        from numba import cuda as ca
        print("  | .. GPU-device: ", str(ca.get_current_device().name), sep="")
        pus = tf.config.experimental.list_physical_devices()
        return "; ".join(["%s (%s)"%(pu.name, pu.device_type) for pu in pus])
    except Exception:
        return "NA"
def getMemoryInfo():
    try:
        from psutil import virtual_memory
        mem = virtual_memory()
        return "%.1f GB"%(mem.total/1000/1000/1000)
    except Exception:
        return "NA"
    
    


print("Initializing PeakBotMRM")
try:
    import platform
    print("  | .. OS:", platform.platform())
except Exception:
    print("  | .. fetching OS information failed")

print("  | .. Python: %s"%(sys.version))
print("  | .. TensorFlow version: %s"%(tf.__version__))

try:
    import cpuinfo
    s = cpuinfo.get_cpu_info()["brand_raw"]
    print("  | .. CPU: %s"%(s))
except Exception:
    print("  | .. fetching CPU info failed")

try:
    from psutil import virtual_memory
    mem = virtual_memory()
    print("  | .. Main memory: %.1f GB"%(mem.total/1000/1000/1000))
except Exception:
    print("  | .. fetching main memory info failed")

try:
    from numba import cuda as ca
    print("  | .. GPU-device: ", str(ca.get_current_device().name), sep="")
    pus = tf.config.experimental.list_physical_devices()
    for pu in pus:
        print("  | .. TensorFlow device: Name '%s', type '%s'"%(pu.name, pu.device_type))
except Exception:
    print("  | .. fetching GPU info failed")
print("")
    




def getDatasetTemplate(templateSize = 1024 * 32, includeMetaInfo = None):
    if includeMetaInfo == None:
        includeMetaInfo = Config.INCLUDEMETAINFORMATION
    template = {"channel.rt"        : np.zeros((templateSize, Config.RTSLICES),   dtype=float),
                "channel.int"       : np.zeros((templateSize, Config.RTSLICES),   dtype=float),
                "inte.peak"         : np.zeros((templateSize, Config.NUMCLASSES), dtype=int),
                "inte.rtStart"      : np.zeros((templateSize),    dtype=float),
                "inte.rtEnd"        : np.zeros((templateSize),    dtype=float),
                "inte.rtInds"       : np.zeros((templateSize, 2), dtype=float),
                "inte.area"         : np.zeros((templateSize),    dtype=float),
                "pred"              : np.zeros((templateSize, Config.NUMCLASSES + 2 + Config.RTSLICES), dtype=float),
               }
    if includeMetaInfo:
        template = {**template, 
                **{"ref.substance" : ["" for i in range(templateSize)],
                   "ref.sample"    : ["" for i in range(templateSize)],
                   "ref.experiment": ["" for i in range(templateSize)],
                   "ref.rt"        : np.zeros((templateSize), dtype=float),
                   "ref.PeakForm"  : ["" for i in range(templateSize)], 
                   "ref.Rt shifts" : ["" for i in range(templateSize)],
                   "ref.Note"      : ["" for i in range(templateSize)],
                   "loss.IOU_Area" : np.ones((templateSize), dtype=float),
                   "ref.criteria"  : ["" for i in range(templateSize)],
                   "ref.polarity"  : ["" for i in range(templateSize)],
                   "ref.type"      : ["" for i in range(templateSize)],
                   "ref.CE"        : ["" for i in range(templateSize)],
                   "ref.CMethod"   : ["" for i in range(templateSize)],
                }
        }
    return template


class Dataset:
    def __init__(self, name = None):
        pass    
    def setName(self, name):
        pass    
    def getElements(self):
        pass    
    def getSizeInformation(self):
        pass
    def addData(self, data):
        pass    
    def getData(self, start, elems = None):
        pass    
    def shuffle(self):
        pass    
    def removeOtherThan(self, start, end):
        pass    
    def split(self, ratio = 0.7):
        pass
    
class MemoryDataset(Dataset):
    def __init__(self, name = None):
        self.data = None
        self.name = name
    
    def setName(self, name):
        self.name = name
        
    def getElements(self):
        if self.data == None:
            return 0
        k = list(self.data.keys())[0]
        temp = self.data[k]
        if isinstance(temp, np.ndarray):
            return temp.shape[0]
        elif isinstance(temp, list):
            return len(temp)
        else:
            raise RuntimeError("Unknonw type for key '%s'"%(k))
            
    def getSizeInformation(self):
        if self.data == None:
            return "no data"
        
        size = 0
        for k in self.data.keys():
            if isinstance(self.data[k], np.ndarray):
                size += self.data[k].itemsize * self.data[k].size  ## bytes
            else:
                size += sizeof(self.data)
                
        return "size of dataset is %.2f MB"%(size / 1000 / 1000)
    
    def addData(self, data):
        if self.data == None:
            self.data = data
        else:
            for k in data.keys():
                if k not in self.data.keys():
                    raise RuntimeError("New key '%s' in new data but not in old data"%(k))
            for k in self.data.keys():
                if k not in data.keys():
                    raise RuntimeError("Key '%s' not present in new data"%(k))
            for k in data.keys():
                if type(data[k]) != type(self.data[k]):
                    raise RuntimeError("Key '%s' in new (%s) and old (%s) data have different types"%(k, type(data[k]), type(self.data[k])))
                if isinstance(data[k], np.ndarray):
                    if len(data[k].shape) == 1:
                        self.data[k] = np.concatenate((self.data[k], data[k]))
                    else:
                        self.data[k] = np.vstack((self.data[k], data[k]))
                elif isinstance(data[k], list): 
                    self.data[k].extend(data[k])
                else:
                    raise RuntimeError("Key '%s' has unknown type"%(k))
    
    def getData(self, start = 0, elems = None):
        if elems == None:
            elems = Config.BATCHSIZE
            
        if start + elems > self.getElements():
            elems = self.getElements() - start
        
        temp = {}
        assertLen = -1
        for k in list(self.data.keys()):
            if isinstance(self.data[k], np.ndarray):
                if len(self.data[k].shape)==1:
                    temp[k] = self.data[k][start:(start+elems)]
                elif len(self.data[k].shape)==2:
                    temp[k] = self.data[k][start:(start+elems),:]
                elif len(self.data[k].shape)==3:
                    temp[k] = self.data[k][start:(start+elems),:,:]
                elif len(self.data[k].shape)==4:
                    temp[k] = self.data[k][start:(start+elems),:,:,:]
                else:
                    raise RuntimeError("Too large np.ndarray")
                if assertLen < 0: assertLen = self.data[k].shape[0]
                assert assertLen == self.data[k].shape[0]

            elif isinstance(self.data[k], list):
                temp[k] = self.data[k][start:(start+elems)]
                if assertLen < 0: assertLen = len(self.data[k])
                assert assertLen == len(self.data[k])
            
            else:
                raise RuntimeError("Unknown class for key '%s'"%(k))
        
        return temp, elems
            
    def shuffle(self):
        if self.data is None:
            return 
                    
        elements = -1
        for k in self.data.keys():
            klen = 0
            if isinstance(self.data[k], np.ndarray):
                klen = self.data[k].shape[0]
            elif isinstance(self.data[k], list):
                klen = len(self.data[k])
            else:
                raise RuntimeError("Unknonw type for key '%s'"%(k))
            
            if elements == -1:
                elements = klen
            if elements != klen:
                raise RuntimeError("Key '%s' has a different number of elements"%(k))
        
        tic()  
        newOrd = None
        for k in self.data.keys():
            if   isinstance(self.data[k], np.ndarray):
                if newOrd is None:
                    newOrd = list(range(self.data[k].shape[0]))
                    random.shuffle(newOrd)
                    
                if len(self.data[k].shape) == 1:
                    self.data[k] = self.data[k][newOrd]
                elif len(self.data[k].shape) == 2:
                    self.data[k] = self.data[k][newOrd,:]
                elif len(self.data[k].shape) == 3:
                    self.data[k] = self.data[k][newOrd,:,:]
                elif len(self.data[k].shape) == 4:
                    self.data[k] = self.data[k][newOrd,:,:,:]
                else:
                    raise RuntimeError("Unknown error 2")

            elif isinstance(self.data[k], list):
                if newOrd is None:
                    newOrd = list(range(len(self.data[k])))
                    random.shuffle(newOrd)
                
                self.data[k] = [self.data[k][i] for i in newOrd]
                
            else:
                raise RuntimeError("Unknwon error 1")
            
    def useOrNotUse(self, useBoolList):
        for k in self.data.keys():
            if   isinstance(self.data[k], np.ndarray):
                if len(self.data[k].shape) == 1:
                    self.data[k] = self.data[k][useBoolList]
                elif len(self.data[k].shape) == 2:
                    self.data[k] = self.data[k][useBoolList,:]
                elif len(self.data[k].shape) == 3:
                    self.data[k] = self.data[k][useBoolList,:,:]
                elif len(self.data[k].shape) == 4:
                    self.data[k] = self.data[k][useBoolList,:,:,:]
                else:
                    raise RuntimeError("Unknown error 2")

            elif isinstance(self.data[k], list):
                self.data[k] = [self.data[k][i] for i, u in enumerate(useBoolList) if u]
            else:
                raise RuntimeError("Unknown error 3")
            
    def removeOtherThan(self, start, end):
        for k in self.data.keys():
            if   isinstance(self.data[k], np.ndarray):
                if len(self.data[k].shape) == 1:
                    self.data[k] = self.data[k][start:end]
                elif len(self.data[k].shape) == 2:
                    self.data[k] = self.data[k][start:end,:]
                elif len(self.data[k].shape) == 3:
                    self.data[k] = self.data[k][start:end,:,:]
                elif len(self.data[k].shape) == 4:
                    self.data[k] = self.data[k][start:end,:,:,:]
                else:
                    raise RuntimeError("Unknown error 2")

            elif isinstance(self.data[k], list):
                self.data[k] = self.data[k][start:end]
            else:
                raise RuntimeError("Unknown error 3")
            
    def split(self, ratio = 0.7):
        splitPos = math.floor(self.getElements() * ratio)
        datA = {}
        datB = {}
        for k in self.data.keys():
            if   isinstance(self.data[k], np.ndarray):
                if len(self.data[k].shape) == 1:
                    datA[k] = self.data[k][:splitPos]
                    datB[k] = self.data[k][splitPos:]
                elif len(self.data[k].shape) == 2:
                    datA[k] = self.data[k][:splitPos,:]
                    datB[k] = self.data[k][splitPos:,:]
                elif len(self.data[k].shape) == 3:
                    datA[k] = self.data[k][:splitPos,:,:]
                    datB[k] = self.data[k][splitPos:,:,:]
                elif len(self.data[k].shape) == 4:
                    datA[k] = self.data[k][:splitPos,:,:,:]
                    datB[k] = self.data[k][splitPos:,:,:,:]
                else:
                    raise RuntimeError("Unknown error 2")

            elif isinstance(self.data[k], list):
                datA[k] = self.data[k][:splitPos]
                datB[k] = self.data[k][splitPos:]
                
        dataSetA = MemoryDataset()
        dataSetA.data = datA
        dataSetB = MemoryDataset()
        dataSetB.data = datB
        
        return dataSetA, dataSetB
        





#####################################
### Data generator methods
### Read files from a directory and prepare them
### for PeakBotMRM training and prediction
##
def getMaxNumberOfEpochsForDataset(dataset, batchSize = None, verbose=False):
    if batchSize is None:
        batchSize = Config.BATCHSIZE
        
    return math.floor(dataset.data["channel.rt"].shape[0]/batchSize)


def modelAdapterGenerator(dataset, xKeys, yKeys, batchSize = None, verbose=False):
    if batchSize is None:
        batchSize = Config.BATCHSIZE
        
    cur = 0
    while cur < dataset.data["channel.rt"].shape[0] and cur + batchSize <= dataset.data["channel.rt"].shape[0]:
        temp, elems = dataset.getData(cur, batchSize)
        x = dict((xKeys[k],v) for k,v in temp.items() if k in xKeys.keys())
        y = dict((yKeys[k],v) for k,v in temp.items() if k in yKeys.keys())

        yield x, y
        cur = cur + batchSize

def modelAdapterTrainGenerator(dataset, verbose=False):
    temp = modelAdapterGenerator(dataset, 
                                 {"channel.int":"channel.int"}, 
                                 {"pred": "pred", "inte.peak":"pred.peak", "inte.rtInds": "pred.rtInds"},
                                 verbose = verbose)
    return temp

def modelAdapterPredictGenerator(dataset, verbose=False):
    temp = modelAdapterGenerator(dataset, 
                                 {"channel.int":"channel.int"}, 
                                 {}, 
                                 verbose = verbose)
    return temp

def convertDatasetToPlain(dataSet, xKeys, yKeys):
    x = {}
    y = {}
    for k, v in xKeys.items():
        x[v] = dataSet.data[k]
    for k, v in yKeys.items():
        y[v] = dataSet.data[k]
    
    return x, y

def convertValDatasetToPlain(dataset):
    return convertDatasetToPlain(dataset, {"channel.int":"channel.int"}, {"pred": "pred", "inte.peak":"pred.peak", "inte.rtInds": "pred.rtInds"})

## taken from MPA, https://stackoverflow.com/q/55237112 (3.6.2022 11:18)
def gaussian_kernel(size, mean, std):
    d = tfp.distributions.Normal(tf.cast(mean, tf.float32), tf.cast(std, tf.float32))
    vals = d.prob(tf.range(start=-size, limit=size+1, dtype=tf.float32))
    return vals / tf.reduce_sum(vals)
    
def gaussian_filter(input, sigma):
    size = int(2*sigma + 0.5)
    kernel = gaussian_kernel(size=size, mean=0.0, std=sigma)
    kernel = kernel[:, tf.newaxis, tf.newaxis]
    conv = tf.nn.conv1d(input, kernel, stride=1, padding="SAME")
    return conv

def diff1d(signal):
    s1 = signal[:,  :-1, :]
    s2 = signal[:, 1:  , :]
    return tf.concat([tf.zeros([tf.shape(signal)[0], 1, signal.shape[2]], dtype = signal.dtype), s2 - s1], axis = 1)


class PeakBotMRM():
    def __init__(self, name, ):
        super(PeakBotMRM, self).__init__()

        batchSize = Config.BATCHSIZE
        rts = Config.RTSLICES
        numClasses = Config.NUMCLASSES
        version = Config.VERSION

        self.name       = name
        self.batchSize  = batchSize
        self.rts        = rts
        self.numClasses = numClasses
        self.model      = None
        self.version    = version


    @timeit
    def buildTFModel(self, mode="predict", verbose = False):
        assert mode in ["training", "predict"], "Incorrect mode ('%s') provided. This must be one of ['training', 'predict']"%(mode)
        
        dropOutRate = Config.DROPOUT
        uNetLayerSizes = Config.UNETLAYERSIZES
        
        inputs = []
        outputs = []

        if verbose:
            print("  | PeakBotMRM v %s model"%(self.version))
            print("  | .. Desc: Detection of a single LC-HRMS peak in an EIC (peak apeax is approaximately in the center of the EIC)")
            print("  | ")

        ## Input: Only LC-HRMS area
        eic = tf.keras.Input(shape=(Config.RTSLICES), name="channel.int")
        inputs.append(eic)
        eicValMin = tf.expand_dims(tf.math.reduce_min(eic, axis=1), axis=-1)
        eicValMax = tf.expand_dims(tf.math.reduce_max(eic, axis=1), axis=-1)
        
        if verbose:
            print("  | .. Inputs")
            print("  | .. .. channel.int is", eic)
            print("  |")
        
        
        if True:
            ## Normalize and scale EIC (remove constant baseline and scale to a maximum intensity value of 1)
            minVal = eicValMin
            minVal = tf.repeat(minVal, repeats=[Config.RTSLICES], axis=1)
            eic = tf.math.subtract(eic, minVal)

            maxVal = tf.where(eicValMax == 0, tf.ones_like(eicValMax), eicValMax)
            maxVal = tf.repeat(maxVal, repeats=[Config.RTSLICES], axis=1)
            eic = tf.math.divide(eic, maxVal)
            
            ## add "virtual" channel to the EICs as required by the convolutions
            eic = tf.expand_dims(eic, axis=-1)
            
            if verbose:            
                print("  | .. Preprocessing")
                print("  | .. .. normalization and scaling (for each standardized EIC: 1) subtraction of minimum value; 2) division by maximum value")
        
        if True:
            ## Feature engineering: derive derivatives of the EICs and use them as 'derived' input features
            news = eic
            
            for sigma in [0.5, 1, 2, 3]:
                eicSmoothed = gaussian_filter(eic, sigma=sigma)
                eicDerivative = diff1d(eicSmoothed)

                if verbose:            
                    print("  | .. .. smoothing EIC with guassian distribution (sigma %f)"%sigma)
                    print("  | .. .. calculated derivative for smoothed signal (padding with 0 in front)")
                
                news = tf.concat([news, eicDerivative], -1)
            
            eic = news
            
            if verbose:            
                print("  | .. .. derived input is", eic)
                print("  |")
        
        x = eic
        for i in range(len(uNetLayerSizes)):
            for width in [2, 1]:
                x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (2*width+1), padding="same", use_bias=False, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                
            x = tf.keras.layers.MaxPooling1D((2))(x)
            x = tf.keras.layers.Dropout(dropOutRate)(x)

        ## Intermediate layer
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        
        ## Predictions
        fx = x
        ## Result type
        peaks  = tf.keras.layers.Dense(Config.NUMCLASSES, activation="sigmoid", name="pred.peak")(fx)
        outputs.append(peaks)
        
        ## Peak borders
        rtInds = tf.keras.layers.Dense(2, activation="relu", name="pred.rtInds")(fx)
        outputs.append(rtInds)
    
        pred = tf.keras.layers.Concatenate(axis=1, name="pred")([peaks, rtInds])
        outputs.append(pred)
        
        if verbose:
            print("  | .. Unet layers are")
            print("  | .. .. [%s]"%(", ".join(str(u) for u in uNetLayerSizes)))
            print("  |")
            print("  | .. Intermediate layer")
            print("  | .. .. flattened layer is", fx)
            print("  | .. ")
            print("  | .. Outputs")
            print("  | .. .. pred.peak   is", peaks)
            print("  | .. .. pred.rtInds is", rtInds)
            print("  | .. .. pred        is", pred)
            print("  | .. ")
            print("  | ")

        self.model = tf.keras.models.Model(inputs, outputs)
        
        losses = {
            "pred.peak": "CategoricalCrossentropy", 
            "pred.rtInds": None, 
            "pred": EICIOULoss } # MSERtInds EICIOULoss
        lossWeights = {
            "pred.peak": 1, 
            "pred.rtInds": None, 
            "pred": 1/10}
        metrics = {
            "pred.peak": ["categorical_accuracy", tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2), Accuracy4Peaks(), Accuracy4NonPeaks()],
            "pred.rtInds": ["MSE"], 
            "pred": [EICIOUPeaks()] }
        
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = Config.LEARNINGRATESTART),
                           loss = losses, 
                           loss_weights = lossWeights, 
                           metrics = metrics)

    @timeit
    def train(self, datTrain, datVal, epochs = None, logDir = None, callbacks = None, verbose = True):
        
        steps_per_epoch = Config.STEPSPEREPOCH
        if epochs is None:
            epochs = Config.EPOCHS * steps_per_epoch
        epochs = math.floor(epochs / steps_per_epoch)

        if verbose:
            print("  | Fitting model on training data")
            print("  | .. Logdir is '%s'"%logDir)
            print("  | .. Number of epochs %d"%(epochs))
            print("  |")

        if logDir is None:
            logDir = os.path.join("logs", "fit", "PeakBotMRM_v" + self.version + "_" + uuid.uuid4().hex)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logDir, histogram_freq = 1, write_graph = True)
        file_writer = tf.summary.create_file_writer(os.path.join(logDir, "scalars"))
        file_writer.set_as_default()

        _callBacks = []
        _callBacks.append(tensorboard_callback)
        if callbacks is not None:
            if type(callbacks) is list:
                _callBacks.extend(callbacks)
            else:
                _callBacks.append(callbacks)

        # Fit model
        print(getHeader("Tensorflow fit function start"))
        history = self.model.fit(
            datTrain,
            validation_data = datVal,

            batch_size      = self.batchSize,
            epochs          = epochs,
            steps_per_epoch = steps_per_epoch,

            callbacks = _callBacks,

            verbose = verbose
        )
        print(getHeader("Tensorflow fit function end"))
        print("")

        return history

    def loadFromFile(self, modelFile):
        self.model.load_weights(modelFile)

    def saveModelToFile(self, modelFile):
        self.model.save_weights(modelFile)







@timeit
def trainPeakBotMRMModel(trainDataset, logBaseDir, modelName = None, valDataset = None, addValidationDatasets = None, everyNthEpoch = -1, epochs = -1, verbose = False):
    tic("pbTrainNewModel")

    ## name new model
    if modelName is None:
        modelName = "%s%s__%s"%(Config.NAME, Config.VERSION, uuid.uuid4().hex[0:6])

    ## log information to folder
    logDir = os.path.join(logBaseDir, modelName)
    logger = tf.keras.callbacks.CSVLogger(os.path.join(logDir, "clog.tsv"), separator="\t")

    if verbose:
        print("Training new PeakBotMRM model")
        print("  | Model name is '%s'"%(modelName))
        print("  | .. config is")
        print(Config.getAsStringFancy().replace(";", "\n"))
        print("  |")

    ## define learning rate schedule
    def lrSchedule(epoch, lr):
        lr = lr / (1 + Config.LEARNINGRATEMULTIPLIER)
        tf.summary.scalar('learningRate', data=lr, step=epoch)

        return max(lr, Config.LEARNINGRATEMINVALUE)
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(lrSchedule, verbose=False)

    ## create generators for training data
    datGenTrain = modelAdapterTrainGenerator(trainDataset, verbose = verbose)
    if epochs == -1:
        epochs = getMaxNumberOfEpochsForDataset(trainDataset)
    if epochs < getMaxNumberOfEpochsForDataset(trainDataset):
        raise RuntimeError("Too few training examples (%d) provided for %d epochs"%(getMaxNumberOfEpochsForDataset(trainDataset), epochs))
    if verbose:
        print("  | .. There are %d training batches available (%s)"%(epochs, trainDataset.getSizeInformation()))
        print("  |")
    
    ## create generators for validation data
    datGenVal   = None
    if valDataset is not None:
        datGenVal = modelAdapterTrainGenerator(valDataset, verbose = verbose)
        datGenVal = tf.data.Dataset.from_tensors(next(datGenVal))

    ## add additional validation datasets to monitor model performance during the trainging process
    valDS = AdditionalValidationSets(logDir,
                                     batch_size=Config.BATCHSIZE,
                                     everyNthEpoch = everyNthEpoch,
                                     verbose = verbose)
    if addValidationDatasets is not None:
        if verbose:
            print("  | Additional validation datasets")
        for valDataset in addValidationDatasets:
            tic("addDS")
            print("  | ..", valDataset.name)
            datGen  = modelAdapterTrainGenerator(valDataset)
            x, y = convertValDatasetToPlain(valDataset)
                            
            if x is not None and y is not None:
                valDS.addValidationSet((x,y, valDataset.name))
                if verbose:
                    print("  | .. .. %d instances; %s"%(x["channel.int"].shape[0], valDataset.getSizeInformation()))

            else:
                raise RuntimeError("Unknonw additional validation dataset")
        if verbose:
            print("  |")

    ## instanciate a new model and set its parameters
    pb = PeakBotMRM(modelName)
    pb.buildTFModel(mode="training", verbose = verbose)

    ## train the model
    history = pb.train(
        datTrain = datGenTrain,
        datVal   = datGenVal,
        epochs   = epochs,

        logDir = logDir,
        callbacks = [logger, lrScheduler, valDS],

        verbose = verbose * 2
    )


    ## save metrices of the training process in a user-convenient format (pandas table)
    metricesAddValDS = None
    if addValidationDatasets is not None:
        hist = valDS.history[-1]
        for valDataset in addValidationDatasets:
            for metric, metName in {"loss":"loss", "pred.peak_MatthewsCorrelationCoefficient":"MCC", "pred_EICIOUPeaks":"Area IOU", "pred.peak_Acc4Peaks": "Sensitivity (peaks)", "pred.peak_Acc4NonPeaks": "Specificity (no peaks)", "pred.peak_categorical_accuracy": "Accuracy"}.items():
                val = hist[valDataset.name + "_" + metric]
                newRow = pd.DataFrame({"model": modelName, "set": valDataset.name, "metric": metName, "value": val}, index=[0])
                if metricesAddValDS is None:
                    metricesAddValDS = newRow
                else:
                    metricesAddValDS = pd.concat((metricesAddValDS, newRow), axis=0, ignore_index=True)
    if verbose:
        print("  |")
        print("  | .. model built and trained successfully (took %.1f seconds)"%toc("pbTrainNewModel"))

    return pb, metricesAddValDS, modelName







@timeit
def integrateArea(eic, rts, start, end):
    method = Config.INTEGRATIONMETHOD
        
    startInd = arg_find_nearest(rts, start)
    endInd   = arg_find_nearest(rts, end)

    if end <= start:
        warnings.warn("Warning in peak area calculation: start and end rt of peak are incorrect (start %.2f, end %.2f). An area of 0 will be returned."%(start, end), RuntimeWarning)
        return 0

    area = 0
    if method.lower() in ["linearbetweenborders", "linear"]:
        ## TODO something is wrong here
        if (rts[endInd]-rts[startInd]) == 0:
            warnings.warn("Warning in peak area calculation: division by 0 (startInd %.2f, endInd %.2f). An area of 0 will be returned."%(startInd, endInd), RuntimeWarning)
            return 0

        minV = min(eic[startInd], eic[endInd])
        maxV = max(eic[startInd], eic[endInd])
        for i in range(startInd, endInd+1):
            area = area + eic[i] - ((rts[i]-rts[startInd])/(rts[endInd]-rts[startInd]) * (maxV - minV) + minV)

    elif method.lower() in ["all"]:
        for i in range(startInd, endInd+1):
            area = area + eic[i]

    elif method.lower() in ["minbetweenborders"]:
        ## replaced with area integration rather than sum of signals
        
        minV = 0
        if endInd > startInd: 
            minV = np.min(eic[startInd:(endInd+1)])
            y = eic[startInd : (endInd + 1)]
            x = rts[startInd : (endInd + 1)] * 60
            area = np.sum((y[1:] + y[: -1] - minV * 2) / 2 * np.diff(x))

    else:
        raise RuntimeError("Unknown integration method")

    
    return area



def calcR2(x, y, yhat):
    return r2_score(y, yhat)
    #ybar = np.sum(y)/len(y)
    #ssreg = np.sum((yhat-ybar)**2)
    #sstot = np.sum((y - ybar)**2)
    #return ssreg / sstot



def calibrationRegression(x, y, type = None):
    try:
        if type is None:
            type = Config.CALIBRATIONMETHOD    
        
        if type == "y=k*x+d": 
            x = np.array(x).reshape((-1, 1))
            y = np.array(y)
            
            model = LinearRegression(positive = True)
            model.fit(x, y)
            yhat = model.predict(x)
            
            r2 = calcR2(x, y, yhat)
            
            return model.predict, r2, yhat, (model.intercept_, model.coef_), "y = %f * x + %f"%(model.coef_, model.intercept_)
        
        if type == "y=k*x+d; 1/expConc.":
            x_ = np.array(x).reshape((-1, 1))
            y_ = np.array(y)
            
            model = LinearRegression(positive = True)
            model.fit(x_, y_, np.ones(len(y))/np.array(x))
            yhat = model.predict(x_)
            
            r2 = calcR2(x, y, yhat)
            
            return model.predict, r2, yhat, (model.intercept_, model.coef_[0]), "y = %f * x + %f"%(model.coef_, model.intercept_)

        if type == "y=k*x**2+l*x+d":
            x_ = np.array(x)
            y_ = np.array(y)
            
            coeffs = np.polyfit(x, y, 2)
            model = np.poly1d(coeffs)
            yhat = model(x)
            
            r2 = calcR2(x, y, yhat)
            
            return model, r2, yhat, coeffs, "y = %f * x**2 + %f * x + %f"%(coeffs[0], coeffs[1], coeffs[2])

        if type == "y=k*x**2+l*x+d; 1/expConc.":
            x_ = np.array(x)
            y_ = np.array(y)
            
            coeffs = np.polyfit(x, y, 2, w = np.ones(len(y))/np.array(x))
            model = np.poly1d(coeffs)
            yhat = model(x)
            
            r2 = calcR2(x, y, yhat)
            
            return model, r2, yhat, coeffs, "y = %f * x**2 + %f * x + %f"%(coeffs[0], coeffs[1], coeffs[2])
    
    except Exception as ex:
        print("Exception in linear regression calibrationRegression(x, y, type) with x '%s', y '%s', type '%s'"%(str(x), str(y), str(type)))
        traceback.print_exc()
        raise ex

    raise RuntimeError("Unknown calibration method '%s' specified"%(type))













def loadModel(modelPath, mode, verbose = True):
    pb = PeakBotMRM("")
    pb.buildTFModel(mode=mode, verbose = verbose)
    pb.loadFromFile(modelPath)
    return pb



@timeit
def runPeakBotMRM(instances, modelPath = None, model = None, verbose = True):
    tic("detecting with PeakBotMRM")

    if verbose:
        print("Detecting peaks with PeakBotMRM")
        print("  | .. loading PeakBotMRM model '%s'"%(modelPath))
        if Config.UPDATEPEAKBORDERSTOMIN:
            print("  | .. ATTENTION: peak bounds will be updated to minimum values in the predicted area")

    pb = model
    if model is None and modelPath is not None:
        model = loadModel(modelPath, mode = "predict")
    
    pred = pb.model.predict(instances["channel.int"], verbose = verbose)
    peakTypes = np.argmax(pred[0], axis=1)
    rtStartInds = np.floor(np.array(pred[1][:,0])).astype(int)
    rtEndInds = np.ceil(np.array(pred[1][:,1])).astype(int)
    
    if Config.UPDATEPEAKBORDERSTOMIN:
        for i in range(pred[1].shape[0]):
            if peakTypes[i] == 0 and rtStartInds[i] < rtEndInds[i]:
                eic = np.copy(instances["channel.int"][i])
                eic[:rtStartInds[i]] = 0
                eic[rtEndInds[i]:] = 0
                maxInd = np.argmax(eic)
                
                eic = np.copy(instances["channel.int"][i])
                mVal = np.max(eic)
                eic[:rtStartInds[i]] = mVal * 1.1
                eic[(maxInd-1):] = mVal * 1.1
                minIndL = np.argmin(eic)
                
                eic = np.copy(instances["channel.int"][i])
                mVal = np.max(eic)
                eic[:(maxInd+1)] = mVal * 1.1
                eic[(rtEndInds[i]+1):] = mVal * 1.1
                minIndR = np.argmin(eic)
                
                #print("Updating eic bounds from ", rtStartInds[i], maxInd, rtEndInds[i], " to ", minIndL, minIndR)
                rtStartInds[i] = minIndL
                rtEndInds[i] = minIndR

    return peakTypes, rtStartInds, rtEndInds



@timeit
def evaluatePeakBotMRM(instancesWithGT, modelPath = None, model = None, verbose = True):
    tic("detecting with PeakBotMRM")

    if verbose:
        print("Evaluating peaks with PeakBotMRM")
        print("  | .. loading PeakBotMRM model '%s'"%(modelPath))

    pb = model
    if model is None and modelPath is not None:
        pb = loadModel(modelPath, mode = "training")
    
    x = {"channel.int": instancesWithGT["channel.int"]}
    y = {"pred"  : np.hstack((instancesWithGT["inte.peak"], instancesWithGT["inte.rtInds"], instancesWithGT["channel.int"])), "pred.peak": instancesWithGT["inte.peak"], "pred.rtInds": instancesWithGT["inte.rtInds"]}
    history = pb.model.evaluate(x, y, return_dict = True, verbose = verbose)

    return history


















class Substance:
    def __init__(self, name, Q1, Q3, CE, CEMethod, refRT, peakForm, rtShift, note, polarity, type, criteria, internalStandard, calLevel1Concentration, calSamples, useCalSamples, calibrationMethod, calculateCalibration):
        self.name = name
        self.Q1 = Q1
        self.Q3 = Q3
        self.CE = CE
        self.CEMethod = CEMethod
        self.refRT = refRT
        self.peakForm = peakForm
        self.rtShift = rtShift
        self.note = note
        self.criteria = criteria
        self.polarity = polarity
        self.type = type
        self.internalStandard = internalStandard
        self.calLevel1Concentration = calLevel1Concentration
        self.calSamples = calSamples
        self.useCalSamples = useCalSamples
        self.calibrationMethod = calibrationMethod
        self.calculateCalibration = calculateCalibration
    
    def __str__(self):
        return "%s (Q1 '%s', Q3 '%s', CE '%s', CEMethod '%s', ref.RT %.2f, calLvl1: '%s', calSamples '%s', calculateCalibration '%s')"%(self.name, self.Q1, self.Q3, self.CE, self.CEMethod, self.refRT, self.calLevel1Concentration, self.calSamples, self.calculateCalibration)

class Integration:
    def __init__(self, foundPeak, rtStart, rtEnd, area, chromatogram, other = None):
        self.foundPeak = foundPeak
        self.rtStart = rtStart
        self.rtEnd = rtEnd
        self.area = area
        self.chromatogram = chromatogram
        if other is None:
            other = {}
        self.other = other






def loadTargets(targetFile, excludeSubstances = None, includeSubstances = None, verbose = True, logPrefix = ""):
    if excludeSubstances is None:
        excludeSubstances = []

    ## load targets
    if verbose: 
        print(logPrefix, "Loading targets from file '%s'"%(targetFile))
    headers, substances = readTSVFile(targetFile, header = True, delimiter = "\t", convertToMinIfPossible = True, getRowsAsDicts = True)
    temp = {}
    for substance in substances:
        substance["Name"] = substance["Name"].replace("  (ISTD)", "").replace(" (ISTD)", "").replace("  Results", "").replace(" Results", "")
        if (excludeSubstances is None or substance["Name"] not in excludeSubstances) and (includeSubstances is None or substance["Name"] in includeSubstances):
            inCalSamples = eval(substance["InCalSamples"])
            temp[substance["Name"]] = Substance(substance["Name"].strip(),
                                                substance["Precursor Ion"],
                                                substance["Product Ion"],
                                                substance["Collision Energy"],
                                                substance["Collision Method"], 
                                                substance["RT"],
                                                substance["PeakForm"], 
                                                substance["RT shifts"],
                                                substance["Note"],
                                                substance["Ion Polarity"],
                                                substance["Type"],
                                                substance["Criteria"], 
                                                None if substance["InternalStandard"] == "" else substance["InternalStandard"].strip(),
                                                substance["Concentration"],
                                                inCalSamples,
                                                inCalSamples.keys() if "UseCalSamples" not in substance.keys() or substance["UseCalSamples"] is None or substance["UseCalSamples"] == "" else eval(substance["UseCalSamples"]),
                                                substance["CalibrationMethod"],
                                                eval("'%s'.lower() == 'true'"%(substance["CalculateCalibration"]))
                                               )
    errors = 0
    for substanceName, substance in temp.items():        
        if substance.internalStandard is not None:
            foundIS = False
            for sub2Name, sub2 in temp.items():
                if substance.internalStandard == sub2.name:
                    foundIS = True
                    
            if not foundIS:
                print("\33[91mError: Internal standard '%s' for substance '%s' not in list\33[0m"%(substance.internalStandard, substance.name))
                errors += 1
    if errors > 0:
        print("\33[91mError: One or several internal standards not present\33[0m")
        raise RuntimeError("Error: One or several internal standards not present")
        
    substances = temp 
    
    if verbose:
        print(logPrefix, "  | .. loaded %d substances"%(len(substances)))
        print(logPrefix, "  | .. of these %d have RT shifts"%(sum((1 if substance.rtShift !="" else 0 for substance in substances.values()))))
        print(logPrefix, "  | .. of these %d have abnormal peak forms"%(sum((1 if substance.peakForm != "" else 0 for substance in substances.values()))))
        print(logPrefix)

    return substances



def loadIntegrations(substances, curatedPeaks, verbose = True, logPrefix = ""):
    ## load integrations
    if verbose:
        print(logPrefix, "Loading integrations from file '%s'"%(curatedPeaks))
    headers, integrationData = parseTSVMultiLineHeader(curatedPeaks, headerRowCount=2, delimiter = ",", commentChar = "#", headerCombineChar = "$")
    headers = dict((k.replace("  (ISTD)", "").replace(" (ISTD)", "").replace("  Results", "").replace(" Results", "").strip(), v) for k,v in headers.items())
    foo = set(header[:header.find("$")].strip() for header in headers if not header.startswith("Sample$"))
    
    notUsingSubstances = []
    for substanceName in substances.keys():
        if substanceName not in foo:
            notUsingSubstances.append(substanceName)
    if verbose and len(notUsingSubstances) > 0:
        print(logPrefix, "\033[91m  | .. Not using %d substances from the transition list as these are not in the integration matrix. These substances are: \033[0m'%s'"%(len(notUsingSubstances), "', '".join(natsort.natsorted(notUsingSubstances))))
    
    notUsingSubstances = []
    for substanceName in foo:
        if substanceName not in substances.keys():
            notUsingSubstances.append(substanceName)
    if verbose and len(notUsingSubstances) > 0:
        print(logPrefix, "\033[91m  | .. Not using %d substances from the integration matrix as these are not in the transition list. These substances are: \033[0m'%s'"%(len(notUsingSubstances), "', '".join(natsort.natsorted(notUsingSubstances))))
    
    foo = dict((k, v) for k, v in substances.items() if k in foo)
    if verbose:
        print(logPrefix, "  | .. restricting substances from %d to %d (overlap of substances and integration results)"%(len(substances), len(foo)))
    substances = foo

    ## process integrations
    integrations = {}
    integratedSamples = set()
    totalIntegrations = 0
    foundPeaks = 0
    foundNoPeaks = 0
    for substanceName in substances.keys():
        integrations[substanceName] = {}
        for integration in integrationData:
            sample = integration[headers["Sample$Data File"]].replace(".d", "")
            area = integration[headers["%s$Area"%(substanceName)]]
            try:
                if area == "" or float(area) == 0:
                    integrations[substanceName][sample] = Integration(False, -1, -1, -1, [])
                    foundNoPeaks += 1
                else:
                    integrations[substanceName][sample] = Integration(True, 
                                                                      float(integration[headers["%s$Int. Start"%(substanceName)]]), 
                                                                      float(integration[headers["%s$Int. End"  %(substanceName)]]), 
                                                                      float(integration[headers["%s$Area"      %(substanceName)]]),
                                                                      [])
                    foundPeaks += 1
            except Exception as ex:
                print("Exception, area is", area)
                raise ex
            integratedSamples.add(sample)
            totalIntegrations += 1
    if verbose:
        print(logPrefix, "  | .. parsed %d integrations from %d substances and %d samples."%(totalIntegrations, len(substances), len(integratedSamples)))
        print(logPrefix, "  | .. there are %d areas and %d no peaks"%(foundPeaks, foundNoPeaks))
        print(logPrefix)
    # integrations [['Pyridinedicarboxylic acid Results', 'R100140_METAB02_MCC025_CAL1_20200306', '14.731', '14.731', '0'], ...]

    return substances, integrations


 
def loadChromatograms(substances, integrations, samplesPath, sampleUseFunction = None, loadFromPickleIfPossible = True,
                      allowedMZOffset = 0.05, MRMHeader = None,
                      pathToMSConvert = "msconvert.exe", maxValCallback = None, curValCallback = None, 
                      verbose = True, logPrefix = ""):
    ## load chromatograms
    tic("procChroms")
    
    foundSamples = {}
    
    if MRMHeader is None:
        MRMHeader = Config.MRMHEADER
    
    if verbose:
        print(logPrefix, "Loading chromatograms")
        
    resetIntegrations = False
    if integrations is None:
        resetIntegrations = True
    
    createNewIntegrations = False
    if integrations is None:
        integrations = {}
        createNewIntegrations = True
    
    samples = os.listdir(samplesPath)
    if maxValCallback is not None:
        maxValCallback(len(samples))
    for samplei, sample in enumerate(os.listdir(samplesPath)):
        if curValCallback is not None:
            curValCallback(samplei)
        
        pathsample = os.path.join(samplesPath, sample)
        if os.path.isdir(pathsample) and pathsample.endswith(".d"):
            foundSamples[Path(sample).stem] = {"path": pathsample, "converted": pathsample.replace(".d", ".mzML")}
            
            try:
                d = dictifyXMLFile(os.path.join(pathsample, "AcqData", "sample_info.xml"))
                try:
                    def getAcqTime(di):
                        fields = d["SampleInfo"]["Field"]
                        for f in fields:
                            if f["DisplayName"][0]["_text"] == "Acquisition Time":
                                return f["Value"][0]["_text"]
                    foundSamples[Path(sample).stem]["Acq. Date-Time"] = getAcqTime(d)
                except:
                    pass
                try:
                    def getMethod(di):
                        fields = d["SampleInfo"]["Field"]
                        for f in fields:
                            if f["DisplayName"][0]["_text"] == "Method":
                                return f["Value"][0]["_text"]
                    foundSamples[Path(sample).stem]["Method"] = getMethod(d)
                except:
                    pass
                try:
                    def getInjVol(di):
                        fields = d["SampleInfo"]["Field"]
                        for f in fields:
                            if f["DisplayName"][0]["_text"] == "Inj Vol (µl)":
                                return f["Value"][0]["_text"]
                    foundSamples[Path(sample).stem]["Inj. volume"] = getInjVol(d)
                except:
                    pass
                try:
                    def getDilution(di):
                        fields = d["SampleInfo"]["Field"]
                        for f in fields:
                            if f["DisplayName"][0]["_text"] == "Dilution":
                                return f["Value"][0]["_text"]
                    foundSamples[Path(sample).stem]["Dilution"] = getDilution(d)
                except:
                    pass
                try:
                    def getComment(di):
                        fields = d["SampleInfo"]["Field"]
                        for f in fields:
                            if f["DisplayName"][0]["_text"] == "Comment":
                                return f["Value"][0]["_text"]
                    foundSamples[Path(sample).stem]["Comment"] = getComment(d).strip()
                except:
                    pass
            except:
                pass
            
            if not os.path.isfile(pathsample.replace(".d", ".mzML")):
                cmd = [pathToMSConvert, "-o", samplesPath, "--mzML", "--z", pathsample]
                subprocess.call(cmd)
                if not os.path.isfile(pathsample.replace(".d", ".mzML")):
                    print("Error: Converting the file '%s' failed. Probably msconvert is not registered in your path, please register it. (command is '%s'"%(sample, cmd))
                    sys.exit(-1)
                print(logPrefix, "  | .. sample '%s' is a folder and ends with '.d'. Thus it was converted to '%s' (command: '%s')"%(sample, sample.replace(".d", ".mzML"), cmd))

    samples = [os.path.join(samplesPath, f) for f in os.listdir(samplesPath) if os.path.isfile(os.path.join(samplesPath, f)) and f.lower().endswith(".mzml")]
    usedSamples = set()
    if os.path.isfile(os.path.join(samplesPath, "integrations.pickle")) and loadFromPickleIfPossible:
        with open(os.path.join(samplesPath, "integrations.pickle"), "rb") as fin:
            integrations, usedSamples = pickle.load(fin)
            if verbose:
                print(logPrefix, "  | .. Imported integrations from pickle file '%s'"%(os.path.join(samplesPath, "integrations.pickle")))
    else:
        if verbose:
            print(logPrefix, "  | .. This might take a couple of minutes as all samples/integrations/channels/etc. need to be compared and the current implementation are 4 sub-for-loops")
        
        if curValCallback is not None:
            curValCallback(0)
        samplei = 0
        for sample in tqdm.tqdm(samples, desc="  | .. importing"):
            if curValCallback is not None:
                curValCallback(samplei)
            samplei = samplei + 1
            sampleName = os.path.basename(sample)
            sampleName = sampleName[:sampleName.rfind(".")]
            usedSamples.add(sampleName)

            foundTargets = []
            unusedChannels = []
            run = pymzml.run.Reader(sample, skip_chromatogram = False)
            
            ## get channels from the chromatogram
            allChannels = []
            for i, entry in enumerate(run):
                if isinstance(entry, pymzml.spec.Chromatogram) and entry.ID.startswith("- SRM"):
                    m = re.match(MRMHeader, entry.ID)
                    Q1, Q3, rtstart, rtend = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))

                    assert rtstart < rtend, "Error: start of XIC is not before its end"

                    polarity = None
                    if entry.get_element_by_name("negative scan") is not None:
                        polarity = "negative"
                    elif entry.get_element_by_name("positive scan") is not None:
                        polarity = "positive"

                    collisionEnergy = None
                    if entry.get_element_by_name("collision energy") is not None:
                        collisionEnergy = entry.get_element_by_name("collision energy").get("value", default=None)
                        if collisionEnergy is not None:
                            collisionEnergy = float(collisionEnergy)

                    collisionMethod = None
                    if entry.get_element_by_name("collision-induced dissociation") is not None:
                        collisionMethod = "collision-induced dissociation"
                    if collisionMethod == "collision-induced dissociation":
                        collisionMethod = "CID"

                    rts = np.array([time for time, intensity in entry.peaks()])
                    eic = np.array([intensity for time, intensity in entry.peaks()])
                    chrom = {"rts": rts, "eic": eic}

                    allChannels.append([Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entry.ID, chrom])

            ## merge channels with integration results for this sample
            alreadyPrinted = []
            for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID, chrom) in enumerate(allChannels):
                usedChannel = []
                useChannel = True
                ## test if channel is unique
                for bi, (bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID, bchrom) in enumerate(allChannels):
                    if i != bi:  ## correct, cannot be optimized as both channels (earlier and later) shall not be used in case of a collision
                        if abs(Q1 - bq1) <= allowedMZOffset and abs(Q3 - bq3) <= allowedMZOffset and \
                            ((rtstart <= brtstart <= rtend) or (rtstart <= brtend <= rtend)) and \
                            polarity == bpolarity and \
                            collisionEnergy == bcollisionEnergy and \
                            collisionMethod == bcollisionMethod:
                            useChannel = False
                            unusedChannels.append(entryID)
                            unusedChannels.append(bentryID)
                            if verbose and "%d - %d"%(i, bi) not in alreadyPrinted: 
                                print(logPrefix, "    \033[91mProblematic channel combination found in sample '%s'. Both will be skipped\033[0m"%(sampleName))
                                print(logPrefix, "        * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                print(logPrefix, "        * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))
                                print(logPrefix, )      
                            alreadyPrinted.append("%d - %d"%(i, bi))
                            alreadyPrinted.append("%d - %d"%(bi, i))                      
                
            for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID, chrom) in enumerate(allChannels):
                if entryID not in unusedChannels:
                    ## use channel if it is unique and find the integrated substance(s) for it
                    if useChannel:
                        for substance in substances.values():
                            if abs(substance.Q1 - Q1) < allowedMZOffset and abs(substance.Q3 - Q3) <= allowedMZOffset and \
                                substance.CE == collisionEnergy and substance.CEMethod == collisionMethod and \
                                rtstart <= substance.refRT <= rtend:
                                if createNewIntegrations and substance.name not in integrations.keys():
                                    integrations[substance.name] = {}
                                if createNewIntegrations and sampleName not in integrations[substance.name].keys():
                                    integrations[substance.name][sampleName] = Integration(foundPeak = None, rtStart = None, rtEnd = None, area = None, chromatogram = [])
                                
                                if substance.name in integrations.keys() and sampleName in integrations[substance.name].keys():
                                    foundTargets.append([substance, entry, integrations[substance.name][sampleName]])
                                    usedChannel.append(substance)
                                    integrations[substance.name][sampleName].chromatogram.append(chrom)
        

        with open(os.path.join(samplesPath, "integrations.pickle"), "wb") as fout:
            pickle.dump((integrations, usedSamples), fout)
            if verbose:
                print(logPrefix, "  | .. Stored integrations to '%s/integrations.pickle'"%os.path.join(samplesPath, "integrations.pickle"))
    
    if resetIntegrations:
        for s in integrations.keys():
            for f in integrations[s].keys():
                integrations[s][f].foundPeak = None
                integrations[s][f].rtStart = None
                integrations[s][f].rtEnd = None
                integrations[s][f].area = None
    
    ## Remove chromatograms with ambiguously selected chromatograms
    remSubstancesChannelProblems = set()
    for substance in integrations.keys():
        foundOnce = False
        for sample in integrations[substance].keys():
            if len(integrations[substance][sample].chromatogram) == 1:
                foundOnce = True
                integrations[substance][sample].chromatogram = integrations[substance][sample].chromatogram[0]
                
            elif len(integrations[substance][sample].chromatogram) > 1:
                remSubstancesChannelProblems.add(substance)
                break
            
            elif len(integrations[substance][sample].chromatogram) == 0:
                integrations[substance][sample].chromatogram = None
                
        if not foundOnce:
            remSubstancesChannelProblems.add(substance)
    if len(remSubstancesChannelProblems) > 0:
        if verbose:
            print(logPrefix, "\033[91m  | .. %d substances were not found as the channel selection was ambiguous and will thus not be used further. These substances are: \033[0m'%s'. "%(len(remSubstancesChannelProblems), "', '".join(natsort.natsorted(remSubstancesChannelProblems))))
        for r in remSubstancesChannelProblems:
            del integrations[r]
    for sub in list(integrations.keys()):
        if sub not in substances.keys():
            integrations.pop(sub)
            
    remSamples = set()
    allSamples = set()
    usedSamples = set()
    for substance in list(integrations.keys()):
        for sample in list(integrations[substance].keys()):
            allSamples.add(sample)
            if sampleUseFunction is not None and not sampleUseFunction(sample):
                integrations[substance].pop(sample)
                remSamples.add(sample)
            else:
                usedSamples.add(sample)
                
    if verbose and sampleUseFunction is not None:
        print(logPrefix, "\033[93m  | .. %d (%.1f%%) of %d samples were removed. These are: \033[0m'%s'"%(len(remSamples), len(remSamples)/len(allSamples)*100, len(allSamples), "', '".join(("'%s'"%s for s in natsort.natsorted(remSamples)))))
        print(logPrefix, "\033[93m  | .. The %d remaining samples are: \033[0m'%s'"%(len(usedSamples), "', '".join(usedSamples)))
    
    ## remove all integrations with more than one scanEvent
    referencePeaks = 0
    noReferencePeaks = 0
    useSubstances = []
    for substance in integrations.keys():
        useSub = False
        for sample in integrations[substance].keys():
            if integrations[substance][sample].chromatogram is not None:
                referencePeaks += 1
                useSub = True
            else:
                noReferencePeaks += 1
        if useSub:
            useSubstances.append(substance)
    
    substances = dict((k, v) for k, v in substances.items() if k in useSubstances)
    integrations = dict((k, v) for k, v in integrations.items() if k in useSubstances)
    
    if verbose:
        print(logPrefix, "  | .. There are %d sample.substances with unambiguous chromatograms (and %d sample.substances with ambiguous chromatograms) from %d samples"%(referencePeaks, noReferencePeaks, len(usedSamples)))
    
    if verbose:
        print(logPrefix, "  | .. took %.1f seconds"%(toc("procChroms")))
        print(logPrefix)

    return substances, integrations, foundSamples

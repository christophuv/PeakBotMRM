import logging

from .core import *

import sys
import os
import uuid
import re
import math
import random
import natsort
from pathlib import Path
import subprocess

import tensorflow as tf
tf.get_logger().setLevel('WARNING')
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

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
    VERSION = "0.9.52"

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
    EXTENDBORDERSUNTILINCREMENT = True
    INCLUDEMETAINFORMATION = False
    CALIBRATIONMETHOD = "linear, 1/expConc."
    CALIBRATIONMETHODENFORCENONNEGATIVE = True
    INTEGRATENOISE = True
    INTEGRATENOISE_StartQuantile = 0.5
    INTEGRATENOISE_EndQuantile = 0.5
    
    MRMHEADER = "- SRM SIC Q1=(\d+\.?\d*[eE]?-?\d+) Q3=(\d+\.?\d*[eE]?-?\d+) start=(\d+\.?\d*[eE]?-?\d+) end=(\d+\.?\d*[eE]?-?\d+)"

    @staticmethod
    def getAsStringFancy():
        return "\n  | ..".join([
            "  | .. %s"%(Config.NAME),
            " Version: " + Config.VERSION,
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
            " Extend peak borders until no reduction: '%s'"%Config.EXTENDBORDERSUNTILINCREMENT, 
            " Calibration method: '%s'"%Config.CALIBRATIONMETHOD,
            " Integrate noise: '%s'"%Config.INTEGRATENOISE,
            " Integrate noise start quantile: %.2f"%Config.INTEGRATENOISE_StartQuantile,
            " Integrate noise end quantile: %.2f"%Config.INTEGRATENOISE_EndQuantile,
        ])

    @staticmethod
    def getAsString():
        return ";".join([
            "%s"%(Config.NAME),
            "Version: " + Config.VERSION,
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
            "Extend peak borders until no reduction: '%s'"%Config.EXTENDBORDERSUNTILINCREMENT, 
            "Calibration method: '%s'"%Config.CALIBRATIONMETHOD,
            "Integrate noise: '%s'"%Config.INTEGRATENOISE,
            "Integrate noise start quantile: %.2f"%Config.INTEGRATENOISE_StartQuantile,
            "Integrate noise end quantile: %.2f"%Config.INTEGRATENOISE_EndQuantile,
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
        logging.info("  | .. GPU-device: %s"%(str(ca.get_current_device().name)))
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
    
    


logging.info("Initializing PeakBotMRM")
try:
    import platform
    logging.info("  | .. OS: %s"%(str(platform.platform())))
except Exception:
    logging.warning("  | .. fetching OS information failed")

logging.info("  | .. Python: %s"%(sys.version))
logging.info("  | .. TensorFlow version: %s"%(tf.__version__))

try:
    import cpuinfo
    s = cpuinfo.get_cpu_info()["brand_raw"]
    logging.info("  | .. CPU: %s"%(s))
except Exception:
    logging.warning("  | .. fetching CPU info failed")

try:
    from psutil import virtual_memory
    mem = virtual_memory()
    logging.info("  | .. Main memory: %.1f GB"%(mem.total/1000/1000/1000))
except Exception:
    logging.warning("  | .. fetching main memory info failed")

try:
    from numba import cuda as ca
    logging.info("  | .. GPU-device: ", str(ca.get_current_device().name), sep="")
    pus = tf.config.experimental.list_physical_devices()
    for pu in pus:
        logging.info("  | .. TensorFlow device: Name '%s', type '%s'"%(pu.name, pu.device_type))
except Exception:
    logging.warning("  | .. fetching GPU info failed")
logging.info("")
    




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
        k = list(self.data)[0]
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
        for k in self.data:
            if isinstance(self.data[k], np.ndarray):
                size += self.data[k].itemsize * self.data[k].size  ## bytes
            else:
                size += sizeof(self.data)
                
        return "size of dataset is %.2f MB"%(size / 1000 / 1000)
    
    def addData(self, data):
        if self.data == None:
            self.data = data
        else:
            for k in data:
                if k not in self.data:
                    raise RuntimeError("New key '%s' in new data but not in old data"%(k))
            for k in self.data:
                if k not in data:
                    raise RuntimeError("Key '%s' not present in new data"%(k))
            for k in data:
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
        for k in list(self.data):
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
        for k in self.data:
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
        for k in self.data:
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
        for k in self.data:
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
        for k in self.data:
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
        for k in self.data:
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
        x = dict((xKeys[k],v) for k,v in temp.items() if k in xKeys)
        y = dict((yKeys[k],v) for k,v in temp.items() if k in yKeys)

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
            logging.info("  | PeakBotMRM v %s model"%(self.version))
            logging.info("  | .. Desc: Detection of a single LC-HRMS peak in an EIC (peak apeax is approaximately in the center of the EIC)")
            logging.info("  | ")

        ## Input: Only LC-HRMS area
        eic = tf.keras.Input(shape=(Config.RTSLICES), name="channel.int")
        inputs.append(eic)
        eicValMin = tf.expand_dims(tf.math.reduce_min(eic, axis=1), axis=-1)
        eicValMax = tf.expand_dims(tf.math.reduce_max(eic, axis=1), axis=-1)
        
        if verbose:
            logging.info("  | .. Inputs")
            logging.info("  | .. .. channel.int is", eic)
            logging.info("  |")
        
        
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
                logging.info("  | .. Preprocessing")
                logging.info("  | .. .. normalization and scaling (for each standardized EIC: 1) subtraction of minimum value; 2) division by maximum value")
        
        if True:
            ## Feature engineering: derive derivatives of the EICs and use them as 'derived' input features
            news = eic
            
            for sigma in [0.5, 1, 2, 3]:
                eicSmoothed = gaussian_filter(eic, sigma=sigma)
                eicDerivative = diff1d(eicSmoothed)

                if verbose:            
                    logging.info("  | .. .. smoothing EIC with guassian distribution (sigma %f)"%sigma)
                    logging.info("  | .. .. calculated derivative for smoothed signal (padding with 0 in front)")
                
                news = tf.concat([news, eicDerivative], -1)
            
            eic = news
            
            if verbose:            
                logging.info("  | .. .. derived input is", eic)
                logging.info("  |")
        
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
            logging.info("  | .. Unet layers are")
            logging.info("  | .. .. [%s]"%(", ".join(str(u) for u in uNetLayerSizes)))
            logging.info("  |")
            logging.info("  | .. Intermediate layer")
            logging.info("  | .. .. flattened layer is", fx)
            logging.info("  | .. ")
            logging.info("  | .. Outputs")
            logging.info("  | .. .. pred.peak   is", peaks)
            logging.info("  | .. .. pred.rtInds is", rtInds)
            logging.info("  | .. .. pred        is", pred)
            logging.info("  | .. ")
            logging.info("  | ")

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
            logging.info("  | Fitting model on training data")
            logging.info("  | .. Logdir is '%s'"%logDir)
            logging.info("  | .. Number of epochs %d"%(epochs))
            logging.info("  |")

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
        logging.info(getHeader("Tensorflow fit function start"))
        history = self.model.fit(
            datTrain,
            validation_data = datVal,

            batch_size      = self.batchSize,
            epochs          = epochs,
            steps_per_epoch = steps_per_epoch,

            callbacks = _callBacks,

            verbose = verbose
        )
        logging.info(getHeader("Tensorflow fit function end"))
        logging.info("")

        return history

    def loadFromFile(self, modelFile):
        self.model.load_weights(modelFile)
        ## TODO switch to full model here
        #self.model = tf.keras.models.load_model(modelFile.replace(".h5", "_model.h5"), custom_objects = {"gaussian_kernel": gaussian_kernel, "gaussian_filter": gaussian_filter, 
        #                                                                                                 "diff1d": diff1d, "EICIOULoss": EICIOULoss, 
        #                                                                                                 "Accuracy4Peaks": Accuracy4Peaks, "Accuracy4NonPeaks": Accuracy4NonPeaks,
        #                                                                                                 "EICIOUPeaks": EICIOUPeaks},
        #                                        compile = False)

    def saveModelToFile(self, modelFile):
        self.model.save_weights(modelFile.replace(".h5", "_modelWeights.h5"))
        self.model.save(modelFile.replace(".h5", "_model.h5"))
        







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
        logging.info("Training new PeakBotMRM model")
        logging.info("  | Model name is '%s'"%(modelName))
        logging.info("  | .. config is")
        logging.info(Config.getAsStringFancy().replace(";", "\n"))
        logging.info("  |")

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
        logging.info("  | .. There are %d training batches available (%s)"%(epochs, trainDataset.getSizeInformation()))
        logging.info("  |")
    
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
            logging.info("  | Additional validation datasets")
        for valDataset in addValidationDatasets:
            tic("addDS")
            logging.info("  | ..", valDataset.name)
            datGen  = modelAdapterTrainGenerator(valDataset)
            x, y = convertValDatasetToPlain(valDataset)
                            
            if x is not None and y is not None:
                valDS.addValidationSet((x,y, valDataset.name))
                if verbose:
                    logging.info("  | .. .. %d instances; %s"%(x["channel.int"].shape[0], valDataset.getSizeInformation()))

            else:
                raise RuntimeError("Unknonw additional validation dataset")
        if verbose:
            logging.info("  |")

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
        logging.info("  |")
        logging.info("  | .. model built and trained successfully (took %.1f seconds)"%toc("pbTrainNewModel"))

    return pb, metricesAddValDS, modelName







@timeit
def integrateArea(eic, rts, start, end):
    method = Config.INTEGRATIONMETHOD
        
    startInd = arg_find_nearest(rts, start)
    endInd   = arg_find_nearest(rts, end)
    
    if end <= start:
        logging.warning("Warning in peak area calculation: start and end rt of peak are incorrect (start %.2f, end %.2f). An area of 0 will be returned."%(start, end))
        return 0
    
    ## ## R code to verify integration
    ## ## Change 3rd parameter of seq to simulate differently fast scanning instruments
    ## d = rep(0, 1000)
    ## for(i in 1:1000){
    ## x = seq(-3, 3, 0.01)  ## change here
    ## y = dnorm(x)*20 + abs(rnorm(length(x), mean=0, sd=0.1))
    ## minV = min(y)
    ## d[i] = sum((y[2:length(y)] + y[1:length(y)-1] - minV * 2) / 2 * diff(x))
    ## }
    ## plot(x,y, main = paste0("Mean is ", mean(d), " sd is ", sd(d)))
    ##
    ## ## value of 0.01 mean: 19.80, sd: 0.051
    ## ## value of 0.1  mean: 19.67, sd: 0.150
    ## ## value of 0.2  mean: 19.62, sd: 0.198
    ## ## Assuming that the instrument has approximately the same scan-settings for all targets
    ## ##    there is virtually no difference in the calculated peak areas. 

    area = 0
    if method.lower() in ["linearbetweenborders", "linear"]:
        raise Exception("Function not implemented")

    elif method.lower() in ["all"]:
        
        if endInd > startInd: 
            y = eic[startInd : (endInd + 1)]
            x = rts[startInd : (endInd + 1)] * 60
            area = np.sum((y[1:] + y[: -1]) / 2 * np.diff(x))

    elif method.lower() in ["minbetweenborders"]:
        ## replaced with area integration rather than sum of signals
        
        minV = 0
        if endInd > startInd: 
            minV = np.min(eic[startInd:(endInd+1)])
            y = eic[startInd : (endInd + 1)]
            x = rts[startInd : (endInd + 1)] * 60
            area = np.sum((y[1:] + y[: -1] - minV * 2) / 2 * np.diff(x))

    else:
        raise Exception("Unknown integration method")

    
    return area



def calcR2(x, y, yhat):
    return r2_score(y, yhat)
    #ybar = np.sum(y)/len(y)
    #ssreg = np.sum((yhat-ybar)**2)
    #sstot = np.sum((y - ybar)**2)
    #return ssreg / sstot


def polyFun(x, a, b, c):
    return a*x**2 + b*x + c
def polyFunNoIntercept(x, a, b):
    return a*x**2 + b*x

def calibrationRegression(x, y, type = None):
    try:
        
        use = []
        for i in range(len(x)):
            if not np.isnan(x[i]) and not np.isinf(x[i]) and not np.isnan(y[i]) and not np.isinf(y[i]):
                use.append(i)
                
        if len(use) <2: 
            return None
        
        x = [x[i] for i in range(len(x)) if i in use]
        y = [y[i] for i in range(len(y)) if i in use]
        
        if type is None:
            type = Config.CALIBRATIONMETHOD    
        
        if type == "linear":  
            x = np.array(x).reshape((-1, 1))
            y = np.array(y)
            
            model = LinearRegression(positive = True)
            model.fit(x, y)
            
            if Config.CALIBRATIONMETHODENFORCENONNEGATIVE and model.intercept_ < 0:
                model = LinearRegression(positive = True, fit_intercept = False)
                model.fit(x, y)            
            
            yhat = model.predict(x)
            r2 = calcR2(x, y, yhat)
            
            return model.predict, r2, yhat, (model.intercept_, model.coef_), "y = %f * x + %f"%(model.coef_, model.intercept_)
        
        if type == "linear, 1/expConc.":
            x_ = np.array(x).reshape((-1, 1))
            y_ = np.array(y)
            
            model = LinearRegression(positive = True)
            model.fit(x_, y_, np.ones(len(y))/np.array(y))
            
            if Config.CALIBRATIONMETHODENFORCENONNEGATIVE and model.intercept_ < 0:
                model = LinearRegression(positive = True, fit_intercept = False)
                model.fit(x_, y_, np.ones(len(y))/np.array(y))
            
            yhat = model.predict(x_)            
            r2 = calcR2(x, y, yhat)
            
            return model.predict, r2, yhat, (model.intercept_, model.coef_[0]), "y = %f * x + %f"%(model.coef_, model.intercept_)

        if type == "quadratic":
            x_ = np.array(x)
            y_ = np.array(y)
            
            popt, pcov = curve_fit(polyFun, x, y)
            model = np.poly1d(popt)
            fun = "y = %f * x**2 + %f * x + %f"%(popt[0], popt[1], popt[2])
            
            if False and Config.CALIBRATIONMETHODENFORCENONNEGATIVE and popt[2] < 0:
                popt, pcov = curve_fit(polyFunNoIntercept, x, y)
                model = np.poly1d(popt)
                fun = "y = %f * x**2 + %f * x "%(popt[0], popt[1])
            
            yhat = model(x)            
            r2 = calcR2(x, y, yhat)
            
            return model, r2, yhat, popt, fun

        if type == "quadratic, 1/expConc.":
            x_ = np.array(x)
            y_ = np.array(y)
            
            popt, pcov = curve_fit(polyFun, x, y, sigma = np.ones(len(y))/np.array(y))
            model = np.poly1d(popt)
            fun = "y = %f * x**2 + %f * x + %f"%(popt[0], popt[1], popt[2])
            
            if False and Config.CALIBRATIONMETHODENFORCENONNEGATIVE and popt[2] < 0:
                popt, pcov = curve_fit(polyFunNoIntercept, x, y, sigma = np.ones(len(y))/np.array(y))
                model = np.poly1d(popt)
                fun = "y = %f * x**2 + %f * x "%(popt[0], popt[1])
            
            yhat = model(x)            
            r2 = calcR2(x, y, yhat)
            
            return model, r2, yhat, popt, fun
    
    except Exception as ex:
        logging.error("Exception in linear regression calibrationRegression(x, y, type) with x '%s', y '%s', type '%s'"%(str(x), str(y), str(type)))
        logging.exception("Exception in calibrationRegression")
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
        logging.info("Detecting peaks with PeakBotMRM")
        logging.info("  | .. loading PeakBotMRM model '%s'"%(modelPath))
        if Config.UPDATEPEAKBORDERSTOMIN:
            logging.info("  | .. ATTENTION: peak bounds will be updated to minimum values in the predicted area")

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
                
                #logging.info("Updating eic bounds from ", rtStartInds[i], maxInd, rtEndInds[i], " to ", minIndL, minIndR)
                rtStartInds[i] = minIndL
                rtEndInds[i] = minIndR

    return peakTypes, rtStartInds, rtEndInds



@timeit
def evaluatePeakBotMRM(instancesWithGT, modelPath = None, model = None, verbose = True):
    tic("detecting with PeakBotMRM")

    if verbose:
        logging.info("Evaluating peaks with PeakBotMRM")
        logging.info("  | .. loading PeakBotMRM model '%s'"%(modelPath))

    pb = model
    if model is None and modelPath is not None:
        pb = loadModel(modelPath, mode = "training")
    
    x = {"channel.int": instancesWithGT["channel.int"]}
    y = {"pred"  : np.hstack((instancesWithGT["inte.peak"], instancesWithGT["inte.rtInds"], instancesWithGT["channel.int"])), "pred.peak": instancesWithGT["inte.peak"], "pred.rtInds": instancesWithGT["inte.rtInds"]}
    history = pb.model.evaluate(x, y, return_dict = True, verbose = verbose)

    return history


















class Substance:
    def __init__(self, name, Q1, Q3, CE, CEMethod, refRT, peakForm, rtShift, note, polarity, type, criteria, internalStandard, calLevel1Concentration, calLevel1ConcentrationUnit, calSamples, calibrationMethod, calculateCalibration, cas = None, inchiKey = None, canSmiles = None):
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
        self.calLevel1ConcentrationUnit = calLevel1ConcentrationUnit
        self.calSamples = calSamples
        self.calibrationMethod = calibrationMethod
        self.calculateCalibration = calculateCalibration
        
        self.cas = cas
        self.inchiKey = inchiKey
        self.canSmiles = canSmiles
    
    def __str__(self):
        return "%s (Q1 '%s', Q3 '%s', CE '%s', CEMethod '%s', ref.RT %.2f, calLvl1: '%s %s', calSamples '%s', calculateCalibration '%s')"%(self.name, self.Q1, self.Q3, self.CE, self.CEMethod, self.refRT, self.calLevel1Concentration, self.calLevel1ConcentrationUnit, self.calSamples, self.calculateCalibration)

class Integration:
    def __init__(self, foundPeak, rtStart, rtEnd, area, chromatogram, type = "Unknown", comment = "", other = None):
        self.type = type
        self.comment = comment
        self.foundPeak = foundPeak
        self.rtStart = rtStart
        self.rtEnd = rtEnd
        self.area = area
        self.istdRatio = None
        self.concentration = None
        self.chromatogram = chromatogram
        if other is None:
            other = {}
        self.other = other






def loadTargets(targetFile, excludeSubstances = None, includeSubstances = None, verbose = True, logPrefix = ""):
    if excludeSubstances is None:
        excludeSubstances = []

    ## load targets
    if verbose: 
        logging.info("%sLoading targets from file '%s'"%(logPrefix, targetFile))
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
                                                substance["Concentration Unit"],
                                                inCalSamples,
                                                substance["CalibrationMethod"],
                                                eval("'%s'.lower() == 'true'"%(substance["CalculateCalibration"])),
                                                
                                                cas = substance["CAS"] if "CAS" in headers else None,
                                                inchiKey = substance["InChIKey"] if "InChIKey" in headers else None,
                                                canSmiles = substance["canonicalSmiles"] if "canonicalSmiles" in headers else None
                                               )
    errors = 0
    for substanceName, substance in temp.items():        
        if substance.internalStandard is not None:
            foundIS = False
            for sub2Name, sub2 in temp.items():
                if substance.internalStandard == sub2.name:
                    foundIS = True
                    
            if not foundIS:
                logging.warning("\33[91mError: Internal standard '%s' for substance '%s' not in list\33[0m"%(substance.internalStandard, substance.name))
                errors += 1
    if errors > 0:
        logging.warning("\33[91mError: One or several internal standards not present\33[0m")
        raise RuntimeError("Error: One or several internal standards not present")
        
    substances = temp 
    
    if verbose:
        logging.info("%s  | .. loaded %d substances"%(logPrefix, len(substances)))
        logging.info("%s  | .. of these %d have RT shifts"%(logPrefix, sum((1 if substance.rtShift !="" else 0 for substance in substances.values()))))
        logging.info("%s  | .. of these %d have abnormal peak forms"%(logPrefix, sum((1 if substance.peakForm != "" else 0 for substance in substances.values()))))
        logging.info(logPrefix)

    return substances



def loadIntegrations(substances, curatedPeaks, delimiter = ",", commentChar = "#", verbose = True, logPrefix = ""):
    ## load integrations
    if verbose:
        logging.info("%sLoading integrations from file '%s'"%(logPrefix, curatedPeaks))
    headers, integrationData = parseTSVMultiLineHeader(curatedPeaks, headerRowCount=2, delimiter = delimiter, commentChar = commentChar, headerCombineChar = "$")
    headers = dict((k.replace("  (ISTD)", "").replace(" (ISTD)", "").replace("  Results", "").replace(" Results", "").strip(), v) for k,v in headers.items())
    foo = set(header[:header.find("$")].strip() for header in headers if not header.startswith("Sample$"))
    
    notUsingSubstances = []
    for substanceName in substances:
        if substanceName not in foo:
            notUsingSubstances.append(substanceName)
    if verbose and len(notUsingSubstances) > 0:
        logging.info("%s\033[91m  | .. Not using %d substances from the transition list as these are not in the integration matrix. These substances are: \033[0m'%s'"%(logPrefix, len(notUsingSubstances), "', '".join(natsort.natsorted(notUsingSubstances))))
    
    notUsingSubstances = []
    for substanceName in foo:
        if substanceName not in substances:
            notUsingSubstances.append(substanceName)
    if verbose and len(notUsingSubstances) > 0:
        logging.info("%s\033[91m  | .. Not using %d substances from the integration matrix as these are not in the transition list. These substances are: \033[0m'%s'"%(logPrefix, len(notUsingSubstances), "', '".join(natsort.natsorted(notUsingSubstances))))
    
    foo = dict((k, v) for k, v in substances.items() if k in foo)
    if verbose:
        logging.info("%s  | .. restricting substances from %d to %d (overlap of substances and integration results)"%(logPrefix, len(substances), len(foo)))
    substances = foo

    ## process integrations
    integrations = {}
    integratedSamples = set()
    totalIntegrations = 0
    foundPeaks = 0
    foundNoPeaks = 0
    for substanceName in substances:
        integrations[substanceName] = {}
        for integration in integrationData:
            sample = integration[headers["Sample$Data File"]].replace(".d", "")
            area = integration[headers["%s$Area"%(substanceName)]]
            
            try:
                if area == "" or float(area) == 0:
                    integrations[substanceName][sample] = Integration(False, -1, -1, -1, [], type = "Reference", comment = "from file '%s'"%(curatedPeaks))
                    foundNoPeaks += 1
                else:
                    integrations[substanceName][sample] = Integration(129, 
                                                                      float(integration[headers["%s$Int. Start"%(substanceName)]]), 
                                                                      float(integration[headers["%s$Int. End"  %(substanceName)]]), 
                                                                      float(integration[headers["%s$Area"      %(substanceName)]]),
                                                                      [], 
                                                                      type = integration[headers["%s$Type"     %(substanceName)]] if "%s$Type" %(substanceName) in headers else "Reference", 
                                                                      comment = "from file '%s'"%(curatedPeaks))
                    foundPeaks += 1
            except Exception as ex:
                logging.error("Exception, area is", area)
                logging.exception()
                raise ex
            integratedSamples.add(sample)
            totalIntegrations += 1
    if verbose:
        logging.info("%s  | .. parsed %d integrations from %d substances and %d samples."%(logPrefix, totalIntegrations, len(substances), len(integratedSamples)))
        logging.info("%s  | .. there are %d areas and %d no peaks"%(logPrefix, foundPeaks, foundNoPeaks))
        logging.info(logPrefix)
    # integrations [['Pyridinedicarboxylic acid Results', 'R100140_METAB02_MCC025_CAL1_20200306', '14.731', '14.731', '0'], ...]

    return substances, integrations

def _getRTOverlap(astart, aend, bstart, bend):
    return (min(aend, bend) - max(astart, bstart)) / (max(aend, bend) - min(astart, bstart))
 
def loadChromatograms(substances, integrations, samplesPath, sampleUseFunction = None, loadFromPickleIfPossible = True,
                      allowedMZOffset = 0.05, MRMHeader = None,
                      pathToMSConvert = "msconvert.exe", maxValCallback = None, curValCallback = None, 
                      verbose = True, logPrefix = "", errorCallback = None):
    ## load chromatograms
    tic("procChroms")
    
    foundSamples = {}
    
    if MRMHeader is None:
        MRMHeader = Config.MRMHEADER
    
    if verbose:
        logging.info("%sLoading chromatograms"%(logPrefix))
        
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
                                return f["Value"][0]["_text"] + " µL"
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
                    foundSamples[Path(sample).stem]["Sample ID"] = getComment(d).strip()
                except:
                    pass
            except:
                pass
            
            if not os.path.isfile(pathsample.replace(".d", ".mzML")):
                cmd = [pathToMSConvert, "-o", samplesPath, "--mzML", "--z", pathsample]
                subprocess.call(cmd)
                if not os.path.isfile(pathsample.replace(".d", ".mzML")):
                    logging.error("Error: Converting the file '%s' failed. Probably msconvert is not registered in your path, please register it. (command is '%s'"%(sample, cmd))
                    if errorCallback is not None:
                        errorCallback("<b>Error converting file '%s'</b><br><br>Please inspect why this file cannot be converted (maybe it is empty) and either convert it yourself or remove it from the analysis. See log for further details. <br><br>For now the file will be skipped."%(pathsample))

    samples = [os.path.join(samplesPath, f) for f in os.listdir(samplesPath) if os.path.isfile(os.path.join(samplesPath, f)) and f.lower().endswith(".mzml")]
    usedSamples = set()
    
    if verbose:
        logging.info("%s  | .. This might take a couple of minutes as all samples/integrations/channels/etc. need to be compared and the current implementation are 4 sub-for-loops"%(logPrefix))
    
    if maxValCallback is not None:
        maxValCallback(len(samples))
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
        alreadyProcessed = []
        for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID, chrom) in enumerate(allChannels):
            ## test if channel is unique
            for bi, (bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID, bchrom) in enumerate(allChannels):
                if i != bi:  ## correct, cannot be optimized as both channels (earlier and later) shall not be used in case of a collision
                    if abs(Q1 - bq1) <= allowedMZOffset and abs(Q3 - bq3) <= allowedMZOffset and \
                                ((rtstart <= brtstart <= rtend) or (rtstart <= brtend <= rtend)) and \
                                polarity == bpolarity and \
                                collisionEnergy == bcollisionEnergy and \
                                collisionMethod == bcollisionMethod and \
                                "%d - %d"%(i, bi) not in alreadyProcessed:
                                    
                        reqForA = []
                        for substance in substances.values():
                            if abs(substance.Q1 - Q1) < allowedMZOffset and abs(substance.Q3 - Q3) <= allowedMZOffset and \
                                substance.CE == collisionEnergy and substance.CEMethod == collisionMethod and \
                                rtstart <= substance.refRT <= rtend:
                                    reqForA.append(substance)
                        reqForB = []
                        for substance in substances.values():
                            if abs(substance.Q1 - bq1) < allowedMZOffset and abs(substance.Q3 - bq3) <= allowedMZOffset and \
                                substance.CE == bcollisionEnergy and substance.CEMethod == bcollisionMethod and \
                                brtstart <= substance.refRT <= brtend:
                                    reqForB.append(substance)
                        
                        if rtstart < brtstart and rtend > brtend:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            if verbose:
                                logging.info("%s   --> Channel A encapsulates Channel B. Channel A will be used and Channel B will be removed"%(logPrefix))
                            unusedChannels.append(bentryID)
                            if verbose:
                                logging.info(logPrefix)
                        
                        elif brtstart < rtstart and brtend > rtend:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            if verbose:
                                logging.info("%s   --> Channel B encapsulates Channel A. Channel B will be used and Channel A will be removed"%(logPrefix))
                            unusedChannels.append(entryID)
                            if verbose:
                                logging.info(logPrefix)
                        
                        elif len(reqForA) == 0 and len(reqForB) == 0:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            if verbose: 
                                logging.info("%s   --> Channel A is not required for any substance. It will be removed."%(logPrefix))
                                logging.info("%s   --> Channel B is not required for any substance. It will be removed."%(logPrefix))
                            unusedChannels.append(entryID)
                            unusedChannels.append(bentryID)
                            if verbose:
                                logging.info(logPrefix)
                            
                        elif len(reqForA) == 1 and len(reqForB) == 0:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            if verbose: 
                                logging.info("%s   --> Channel A is required for '%s' at %.2f min. It will be kept."%(logPrefix, reqForA[0].name, reqForA[0].refRT))
                                logging.info("%s   --> Channel B is unsused. It will be removed."%(logPrefix))
                            unusedChannels.append(bentryID)
                            if verbose:
                                logging.info(logPrefix)
                            
                        elif len(reqForA) == 0 and len(reqForB) == 1:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            if verbose: 
                                logging.info("%s   --> Channel A is unused. It will be removed."%(logPrefix))
                                logging.info("%s   --> Channel B is required for '%s' at %.2f min. It will be kept"%(logPrefix, reqForB[0].name, reqForB[0].refRT))
                            unusedChannels.append(entryID)
                            if verbose:
                                logging.info(logPrefix)
                        
                        elif len(reqForA) == 1 and len(reqForB) == 1 and reqForA[0].name == reqForB[0].name and _getRTOverlap(rtstart, rtend, brtstart, brtend) > 0.95:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            used = "A" if rtstart < brtstart else "B"
                            unused = "B" if used == "A" else "A"
                            if verbose:
                                logging.info("%s   --> The two channels are used for the same substance and their RT starts/ends are virtually identical. Channel %s will be used and Channel %s will be removed"%(logPrefix, used, unused))
                            unusedChannels.append(bentryID if unused == "B" else entryID)
                            if verbose:
                                logging.info(logPrefix)
                        
                        elif len(reqForA) == 1 and len(reqForB) == 1 and reqForA[0].name == reqForB[0].name and _getRTOverlap(rtstart, rtend, brtstart, brtend) <= 0.95:
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                            midA = (rtstart + rtend) / 2
                            midB = (brtstart + brtend) / 2
                            used = "A" if abs(midA - reqForA[0].refRT) < abs(midB - reqForA[0].refRT) else "B"
                            unused = "B" if used == "A" else "A"
                            
                            if verbose:
                                logging.info("%s   --> The two channels are used for the same substance and their RT starts/ends are similar. Channel %s fits the refernce rt of the substance better will be used and Channel %s will be removed"%(logPrefix, used, unused))
                            unusedChannels.append(bentryID if unused == "B" else entryID)
                            if verbose:
                                logging.info(logPrefix)
                            
                        elif len(reqForA) == len(reqForB) and all([i.name in [j.name for j in reqForB] for i in reqForA]):
                            if verbose: 
                                logging.info("%s    \033[91mProblematic channel combination found in sample '%s'.\033[0m"%(logPrefix, sampleName))
                                logging.info("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.info("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))

                                logging.info("%s   --> Channel A is used for %d substances"%(logPrefix, len(reqForA)))
                                for sub in reqForA:
                                    logging.info("%s      * %s at %.2f min"%(logPrefix, sub.name, sub.refRT))
                                logging.info("%s   --> Channel B is used for %d substances"%(logPrefix, len(reqForB)))
                                for sub in reqForB:
                                    logging.info("%s      * %s at %.2f min"%(logPrefix, sub.name, sub.refRT))
                                    
                            used = "A" if rtstart < brtstart else "B"
                            unused = "B" if used == "A" else "B"
                            
                            if verbose: 
                                logging.info("%s   --> Channel %s will be used as it starts a bit earlier. Channel %s will removed"%(logPrefix, used, unused))
                            unusedChannels.append(bentryID if unused == "B" else entryID)
                            
                            if verbose: 
                                logging.info(logPrefix)

                        else:
                            if verbose: 
                                logging.warning("%s    \033[91mProblematic channel combination found in sample '%s'. Both will be skipped\033[0m"%(logPrefix, sampleName))
                                logging.warning("%s       A * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                                logging.warning("%s       B * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionMethod, bentryID))
                                
                                logging.warning("%s   --> Channel A is used for %d substances"%(logPrefix, len(reqForA)))
                                for sub in reqForA:
                                    logging.warning("%s      * %s at %.2f min"%(logPrefix, sub.name, sub.refRT))
                                logging.warning("%s   --> Channel B is used for %d substances"%(logPrefix, len(reqForB)))
                                for sub in reqForB:
                                    logging.warning("%s      * %s at %.2f min"%(logPrefix, sub.name, sub.refRT))
                                    
                                logging.warning("%s       TODO: Problematic. Both channels will be removed for now"%(logPrefix))
                                
                            unusedChannels.append(entryID)
                            unusedChannels.append(bentryID)
                        
                        alreadyProcessed.append("%d - %d"%(i, bi))
                        alreadyProcessed.append("%d - %d"%(bi, i))       
            
        usedChannel = []
        for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID, chrom) in enumerate(allChannels):
            if entryID not in unusedChannels:
                ## use channel if it is unique and find the integrated substance(s) for it
                for substance in substances.values():
                    if abs(substance.Q1 - Q1) < allowedMZOffset and abs(substance.Q3 - Q3) <= allowedMZOffset and \
                        substance.CE == collisionEnergy and substance.CEMethod == collisionMethod and \
                        rtstart <= substance.refRT <= rtend:
                        if createNewIntegrations and substance.name not in integrations:
                            integrations[substance.name] = {}
                        if createNewIntegrations and sampleName not in integrations[substance.name]:
                            integrations[substance.name][sampleName] = Integration(foundPeak = None, rtStart = None, rtEnd = None, area = None, type = "None", comment = "None", chromatogram = [])
                        
                        if substance.name in integrations and sampleName in integrations[substance.name]:
                            foundTargets.append([substance, entry, integrations[substance.name][sampleName]])
                            usedChannel.append(substance)
                            integrations[substance.name][sampleName].chromatogram.append(chrom)
                            if verbose: 
                                logging.info("%s       Channel will be used for '%s': * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, substance.name, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                
            elif verbose: 
                logging.info("%s       Channel will not be used: * Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, Polarity '%10s', Fragmentation %5.1f '%s', Header '%s'"%(logPrefix, Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionMethod, entryID))
                
    if resetIntegrations:
        for s in integrations:
            for f in integrations[s]:
                integrations[s][f].foundPeak = None
                integrations[s][f].rtStart = None
                integrations[s][f].rtEnd = None
                integrations[s][f].area = None
    
    ## Remove chromatograms with ambiguously selected chromatograms
    remSubstancesChannelProblems = set()
    for substance in integrations:
        foundOnce = False
        for sample in integrations[substance]:
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
            logging.info("%s\033[91m  | .. %d substances were not found as the channel selection was ambiguous and will thus not be used further. These substances are: \033[0m'%s'. "%(logPrefix, len(remSubstancesChannelProblems), "', '".join(natsort.natsorted(remSubstancesChannelProblems))))
        for r in remSubstancesChannelProblems:
            del integrations[r]
    for sub in list(integrations):
        if sub not in substances:
            integrations.pop(sub)
            
    remSamples = set()
    allSamples = set()
    usedSamples = set()
    for substance in list(integrations):
        for sample in list(integrations[substance]):
            allSamples.add(sample)
            if sampleUseFunction is not None and not sampleUseFunction(sample):
                integrations[substance].pop(sample)
                remSamples.add(sample)
            else:
                usedSamples.add(sample)
                
    if verbose and sampleUseFunction is not None:
        logging.info("%s\033[93m  | .. %d (%.1f%%) of %d samples were removed. These are: \033[0m'%s'"%(logPrefix, len(remSamples), len(remSamples)/len(allSamples)*100, len(allSamples), "', '".join(("'%s'"%s for s in natsort.natsorted(remSamples)))))
        logging.info("%s\033[93m  | .. The %d remaining samples are: \033[0m'%s'"%(logPrefix, len(usedSamples), "', '".join(usedSamples)))
    
    ## remove all integrations with more than one scanEvent
    referencePeaks = 0
    noReferencePeaks = 0
    useSubstances = []
    for substance in integrations:
        useSub = False
        for sample in integrations[substance]:
            if integrations[substance][sample].chromatogram is not None:
                referencePeaks += 1
                useSub = True
            else:
                noReferencePeaks += 1
        if useSub:
            useSubstances.append(substance)
    
    if verbose:
        for substance in substances:
            if substance not in useSubstances:
                logging.info("%s\033[91m  | .. Substance '%s' not found as any channel and will not be used\033[0m. "%(logPrefix, substance))
            
    substances = dict((k, v) for k, v in substances.items() if k in useSubstances)
    integrations = dict((k, v) for k, v in integrations.items() if k in useSubstances)
    
    if verbose:
        logging.info("%s  | .. There are %d sample.substances with unambiguous chromatograms (and %d sample.substances with ambiguous chromatograms) from %d samples"%(logPrefix, referencePeaks, noReferencePeaks, len(usedSamples)))
    
    if verbose:
        logging.info("%s  | .. took %.1f seconds"%(logPrefix, toc("procChroms")))
        logging.info(logPrefix)

    return substances, integrations, foundSamples

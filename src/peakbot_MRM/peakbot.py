import logging

from numba.core.types.functions import NumberClass

from .core import *

import sys
import os
import pickle
import uuid
import re

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

import tqdm
import pymzml




#####################################
### Configuration class
##
class Config(object):
    """Base configuration class"""

    NAME    = "PeakBot_MRM"
    VERSION = "0.9"

    RTSLICES       = 256   ## should be of 2^n
    NUMCLASSES     =   2   ## [isFullPeak, hasCoelutingPeakLeftAndRight, hasCoelutingPeakLeft, hasCoelutingPeakRight, isWall, isBackground]
    FIRSTNAREPEAKS =   1   ## specifies which of the first n classes represent a chromatographic peak (i.e. if classes 0,1,2,3 represent a peak, the value for this parameter must be 4)

    BATCHSIZE     =  16#2# 16
    STEPSPEREPOCH =  8#4#  8
    EPOCHS        =  300#7#110

    DROPOUT        = 0.2
    UNETLAYERSIZES = [32,64,128,256]

    LEARNINGRATESTART              = 0.005
    LEARNINGRATEDECREASEAFTERSTEPS = 5
    LEARNINGRATEMULTIPLIER         = 0.9
    LEARNINGRATEMINVALUE           = 3e-7

    INSTANCEPREFIX = "___PBsample_"

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
        ])



print("Initializing PeakBot_MRM")
try:
    import platform
    print("  | .. OS:", platform.platform())
except Exception:
    print("  | .. fetching OS information failed")

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

    gpus = tf.config.experimental.list_physical_devices()
    for gpu in gpus:
        print("  | .. TensorFlow device: Name '%s', type '%s'"%(gpu.name, gpu.device_type))
except Exception:
    print("  | .. fetching GPU info failed")
    










#####################################
### Data generator methods
### Read files from a directory and prepare them
### for PeakBot training and prediction
##
def dataGenerator(folder, instancePrefix = None, verbose=False):

    if instancePrefix is None:
        instancePrefix = Config.INSTANCEPREFIX

    ite = 0
    while os.path.isfile(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite))):
        l = pickle.load(open(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite)), "rb"))
        assert all(np.amax(l["channel.int"], (1)) == 1), "EIC is not scaled to a maximum of 1 '%s'"%(str(np.amax(l["channel.int"], (1))))
        yield l
        ite += 1

def modelAdapterGenerator(datGen, xKeys, yKeys, newBatchSize = None, verbose=False):
    ite = 0
    l = next(datGen)
    while l is not None:

        if verbose and ite == 0:
            logging.info("  | Generated data is")

            for k, v in l.items():
                if type(v).__module__ == "numpy":
                    logging.info("  | .. gt: %18s numpy: %30s %10s"%(k, v.shape, v.dtype))
                else:
                    logging.info("  | .. gt: %18s  type:  %40s"%(k, type(v)))
            logging.info("  |")

        if newBatchSize is not None:
            for k in l.keys():
                if   isinstance(l[k], np.ndarray) and len(l[k].shape)==1:
                    l[k] = l[k][0:newBatchSize]
                    
                if   isinstance(l[k], np.ndarray) and len(l[k].shape)==2:
                    l[k] = l[k][0:newBatchSize,:]

                elif isinstance(l[k], np.ndarray) and len(l[k].shape)==3:
                    l[k] = l[k][0:newBatchSize,:,:]

                elif isinstance(l[k], np.ndarray) and len(l[k].shape)==4:
                    l[k] = l[k][0:newBatchSize,:,:,:]

                elif isinstance(l[k], list):
                    l[k] = l[k][0:newBatchSize]

        if "channel.int" in l.keys() and "inte.peak" in l.keys() and "inte.rtInds" in l.keys():
            l["pred"] = np.hstack((l["inte.peak"], l["inte.rtInds"], l["channel.int"]))
        x = dict((xKeys[k],v) for k,v in l.items() if k in xKeys.keys())
        y = dict((yKeys[k],v) for k,v in l.items() if k in yKeys.keys())

        yield x,y
        l = next(datGen)
        ite += 1

def modelAdapterTrainGenerator(datGen, newBatchSize = None, verbose=False):
    temp = modelAdapterGenerator(datGen, 
                                 {"channel.int":"channel.int"}, 
                                 {"pred": "pred", "inte.peak":"pred.peak", "inte.rtInds": "pred.rtInds"},
                                 newBatchSize, verbose = verbose)
    return temp

def modelAdapterPredictGenerator(datGen, newBatchSize = None, verbose=False):
    temp = modelAdapterGenerator(datGen, 
                                 {"channel.int":"channel.int"}, 
                                 {}, 
                                 newBatchSize, verbose = verbose)
    return temp














#####################################
### PeakBot additional methods
##
@tf.autograph.experimental.do_not_convert
def EICIOU(dummyX, dummyY):
    ## separate user integration and eic
    peaks   = dummyX[:, 0:Config.NUMCLASSES]
    rtInds  = dummyX[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    eic     = dummyX[:, (Config.NUMCLASSES + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:Config.NUMCLASSES]
    prtInds = dummyY[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    
    ## Calculate indices for EICs
    indices = tf.transpose(tf.reshape(tf.repeat(tf.range(tf.shape(eic)[1], dtype=eic.dtype), repeats=tf.shape(eic)[0]), [tf.shape(eic)[1], tf.shape(eic)[0]]))

    ## Extract area for user integration
    stripped = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(rtInds[:,0]), repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                            tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (rtInds[:,1]), repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                        eic, tf.zeros_like(eic))
    maxRow = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), [tf.shape(eic)[0], tf.shape(eic)[1]]), stripped)
    minVal = tf.reduce_min(maxRow, axis=1)
    stripped = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), [tf.shape(eic)[0], tf.shape(eic)[1]]))
    stripped = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    inteArea = tf.reduce_sum(stripped, axis=1)

    ## Extract area for PeakBot_MRM integration
    stripped = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(prtInds[:,0]), repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                            tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (prtInds[:,1]), repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                        eic, tf.zeros_like(eic))
    maxRow = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), [tf.shape(eic)[0], tf.shape(eic)[1]]), stripped)
    minVal = tf.reduce_min(maxRow, axis=1)
    stripped = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), [tf.shape(eic)[0], tf.shape(eic)[1]]))
    stripped = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    pbCalcArea = tf.reduce_sum(stripped, axis=1)

    ## Extract area for overlap of user and PeakBot_MRM integration
    beginInds = tf.math.floor(tf.math.maximum(rtInds[:,0], prtInds[:,0]))
    endInds   = tf.math.ceil (tf.math.minimum(rtInds[:,1], prtInds[:,1]))
    stripped  = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(beginInds, repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                             tf.math.less_equal   (indices, tf.reshape(tf.repeat(endInds  , repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                            eic, tf.zeros_like(eic))
    maxRow = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), [tf.shape(eic)[0], tf.shape(eic)[1]]), stripped)
    minVal = tf.reduce_min(maxRow, axis=1)
    stripped = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), [tf.shape(eic)[0], tf.shape(eic)[1]]))
    stripped = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    overlapArea = tf.reduce_sum(stripped, axis=1)
    #overlapArea = tf.where(tf.math.equal(tf.argmax(peaks, axis=1), 1), tf.zeros_like(overlapArea), overlapArea)

    ## Calculate IOU
    iou = tf.divide(overlapArea+0.0001, tf.subtract(tf.add(inteArea, pbCalcArea), overlapArea)+0.0001)
    ## set IOU to 0 if gt and prediction of peak do not match
    iou = tf.where(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), iou, tf.zeros_like(iou))
    ## set IOU to 1 if gt and prediction of peak match and if no peak was detected
    iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 1), tf.ones_like(iou), iou)
    
    return iou

def EICIOUPeaks(dummyX, dummyY):
    ## separate user integration and eic
    peaks   = dummyX[:, 0:Config.NUMCLASSES]
    rtInds  = dummyX[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    eic     = dummyX[:, (Config.NUMCLASSES + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:Config.NUMCLASSES]
    prtInds = dummyY[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]

    ## get IOU
    iou = EICIOU(dummyX, dummyY)

    ## Only get IOU for true peaks
    iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iou, tf.zeros_like(iou))
    
    ## Calculate IOU only for peaks and return a mean value
    return iou   ## TODO only include iou of peaks in this calculation

def EICIOULoss(dummyX, dummyY):
    ## separate user integration and eic
    peaks   = dummyX[:, 0:Config.NUMCLASSES]
    rtInds  = dummyX[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    eic     = dummyX[:, (Config.NUMCLASSES + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:Config.NUMCLASSES]
    prtInds = dummyY[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]

    ## get IOUloss 
    iou = 1 - EICIOU(dummyX, dummyY)

    ## Only get IOUloss for true peaks
    iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iou, tf.zeros_like(iou))

    ## Combine iou loss with MSE
    mse = tf.reduce_mean(tf.square(rtInds-prtInds), axis=1)    
    return mse * tf.sqrt(tf.abs(iou))


@tf.autograph.experimental.do_not_convert
def CCEPeak(dummyX, dummyY):
    ## separate user integration and eic
    peaks   = dummyX[:, 0:Config.NUMCLASSES]
    rtInds  = dummyX[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    eic     = dummyX[:, (Config.NUMCLASSES + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:Config.NUMCLASSES]
    prtInds = dummyY[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
   
    cca = tf.keras.losses.CategoricalCrossentropy()(peaks, ppeaks)
    return cca

@tf.autograph.experimental.do_not_convert
def CCAPeak(dummyX, dummyY):
    ## separate user integration and eic
    peaks   = dummyX[:, 0:Config.NUMCLASSES]
    rtInds  = dummyX[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    eic     = dummyX[:, (Config.NUMCLASSES + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:Config.NUMCLASSES]
    prtInds = dummyY[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
   
    cca = tf.keras.metrics.categorical_accuracy(peaks, ppeaks)
    return cca

@tf.autograph.experimental.do_not_convert
def MSERtInds(dummyX, dummyY):
    ## separate user integration and eic
    peaks   = dummyX[:, 0:Config.NUMCLASSES]
    rtInds  = dummyX[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]
    eic     = dummyX[:, (Config.NUMCLASSES + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:Config.NUMCLASSES]
    prtInds = dummyY[:, Config.NUMCLASSES:(Config.NUMCLASSES + 2)]

    mse = tf.keras.losses.MeanSquaredError()(rtInds, prtInds)
    return mse


def convertGeneratorToPlain(gen, numIters=1):
    x = None
    y = None
    for i, t in enumerate(gen):
        if i < numIters:
            if x is None:
                x = t[0]
                y = t[1]
            else:
                for k in x.keys():
                    x[k] = np.concatenate((x[k], t[0][k]), axis=0)
                for k in y.keys():
                    y[k] = np.concatenate((y[k], t[1][k]), axis=0)
        else:
            break
    
    return x,y

## modified from https://stackoverflow.com/a/47738812
## https://github.com/LucaCappelletti94/keras_validation_sets
class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, logDir, validation_sets=None, verbose=0, batch_size=None, steps=None, everyNthEpoch=1):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.logDir = logDir
        self.validation_sets = []
        if validation_sets is not None:
            for validation_set in validation_sets:
                self.addValidationSet(validation_set)
        self.verbose = verbose
        self.batch_size = batch_size
        self.steps = steps
        self.everyNthEpoch = everyNthEpoch
        self.lastEpochNum = 0
        self.history = []
        self.printWidths = {}
        self.maxLenNames = 0

    def addValidationSet(self, validation_set):
        if len(validation_set) not in [3, 4]:
            raise ValueError()
        self.validation_sets.append(validation_set)


    @timeit
    def on_epoch_end(self, epoch, logs=None, ignoreEpoch = False):
        self.lastEpochNum = epoch
        hist = None
        if (self.everyNthEpoch > 0 and epoch%self.everyNthEpoch == 0) or ignoreEpoch:
            hist={}
            if self.verbose: print("Additional test datasets (epoch %d): "%(epoch+1))
            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
                tic("kl234hlkjsfkjh1hlkjhasfdkjlh")
                outStr = []
                if len(validation_set) == 2:
                    validation_data, validation_set_name = validation_set
                    validation_targets = None
                    sample_weights = None
                if len(validation_set) == 3:
                    validation_data, validation_targets, validation_set_name = validation_set
                    sample_weights = None
                elif len(validation_set) == 4:
                    validation_data, validation_targets, sample_weights, validation_set_name = validation_set
                else:
                    raise ValueError()

                results = self.model.evaluate(x=validation_data,
                                              y=validation_targets,
                                              verbose=False,
                                              sample_weight=sample_weights,
                                              batch_size=self.batch_size,
                                              steps=self.steps)

                self.maxLenNames = max(self.maxLenNames, len(validation_set_name))

                file_writer = tf.summary.create_file_writer(self.logDir + "/" + validation_set_name)
                metNames = self.model.metrics_names
                metVals = results
                if len(metNames) == 1:
                    metVals = [metVals]
                for i, (metric, result) in enumerate(zip(metNames, metVals)):
                    valuename = "epoch_" + metric
                    with file_writer.as_default():
                        tf.summary.scalar(valuename, data=result, step=epoch)
                    if i > 0: outStr.append(", ")
                    valuename = metric
                    if i not in self.printWidths.keys():
                        self.printWidths[i] = 0
                    self.printWidths[i] = max(self.printWidths[i], len(valuename))
                    outStr.append("%s: %.4f"%("%%%ds"%self.printWidths[i]%valuename, result))
                    hist[validation_set_name + "_" + valuename] = result
                outStr.append("")
                outStr.insert(0, "   %%%ds  - %3.0fs - "%(self.maxLenNames, toc("kl234hlkjsfkjh1hlkjhasfdkjlh"))%validation_set_name)
                if self.verbose: print("".join(outStr))
            if self.verbose: print("")
        self.history.append(hist)

    def on_train_begin(self, logs=None):
        self.on_epoch_end(self.lastEpochNum, logs = logs, ignoreEpoch = True)

    def on_train_end(self, logs=None):
        self.on_epoch_end(self.lastEpochNum, logs = logs, ignoreEpoch = True)

class PeakBot():
    def __init__(self, name, ):
        super(PeakBot, self).__init__()

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
            print("  | PeakBot_MRM v %s model"%(self.version))
            print("  | .. Desc: Detection of a single LC-HRMS peak in an area")
            print("  | ")

        ## Input: Only LC-HRMS area
        eic = tf.keras.Input(shape=(self.rts, 1), name="channel.int")
        inputs.append(eic)
        
        if verbose:
            print("  | .. Inputs")
            print("  | .. .. channel.int is", eic)
            print("  |")

        ## Encoder
        x = eic
        cLayers = [x]
        for i in range(len(uNetLayerSizes)):

            x = tf.keras.layers.ZeroPadding1D(padding=2)(x)
            x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (5), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Dropout(dropOutRate)(x)
            
            x = tf.keras.layers.ZeroPadding1D(padding=1)(x)
            x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (3), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool1D((2))(x)
            cLayers.append(x)
        lastUpLayer = x

        ## Intermediate layer and feature properties (indices and borders)
        x      = tf.keras.layers.BatchNormalization()(x)
        fx     = tf.keras.layers.Flatten()(x)
        
        peaks  = tf.keras.layers.Dense(Config.NUMCLASSES, activation="sigmoid", name="pred.peak")(fx)
        outputs.append(peaks)
        rtInds = tf.keras.layers.Dense(2, activation="relu", name="pred.rtInds")(fx)
        outputs.append(rtInds)
        
        pred = tf.keras.layers.Concatenate(axis=1, name="pred")([peaks, rtInds])#tf.keras.layers.Dense(Config.NUMCLASSES + 2 + Config.RTSLICES, name = "pred", activation = "sigmoid")(fx)
        outputs.append(pred)
        
        if verbose:
            print("  | .. Intermediate layer")
            print("  | .. .. lastUpLayer is", lastUpLayer)
            print("  | .. .. fx          is", fx)
            print("  | .. ")
            print("  | .. Outputs")
            print("  | .. .. pred.peak   is", peaks)
            print("  | .. .. pred.rtInds is", rtInds)
            print("  | .. .. pred        is", pred)
            print("  | .. ")
            print("  | ")

        self.model = tf.keras.models.Model(inputs, outputs)
        
        losses      = {"pred.peak": "CategoricalCrossentropy", "pred.rtInds": None, "pred": EICIOULoss} # MSERtInds EICIOULoss
        lossWeights = {"pred.peak": 1                        , "pred.rtInds": None, "pred": 1/200         }
        metrics     = {"pred.peak": ["categorical_accuracy", tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)]   , "pred.rtInds": "MSE", "pred": [EICIOU, EICIOUPeaks]}
        
        #losses      = {"pred.peak": "CategoricalCrossentropy", "pred.rtInds": "MSE", "pred": None}
        #lossWeights = {"pred.peak": 1                        , "pred.rtInds": 1/200, "pred": None     }
        #metrics     = {"pred.peak": ["categorical_accuracy", tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)]   , "pred.rtInds": None,  "pred": EICIOULoss}
        
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = Config.LEARNINGRATESTART),
                           loss = losses, loss_weights = lossWeights, metrics = metrics)

    @timeit
    def train(self, datTrain, datVal, logDir = None, callbacks = None, verbose = True):
        epochs = Config.EPOCHS
        steps_per_epoch = Config.STEPSPEREPOCH

        if verbose:
            print("  | Fitting model on training data")
            print("  | .. Logdir is '%s'"%logDir)
            print("  | .. Number of epochs %d"%(epochs))
            print("  |")

        if logDir is None:
            logDir = os.path.join("logs", "fit", "PeakBot_v" + self.version + "_" + uuid.uuid4().hex)

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
        history = self.model.fit(
            datTrain,
            validation_data = datVal,

            batch_size      = self.batchSize,
            epochs          = epochs,
            steps_per_epoch = steps_per_epoch,

            callbacks = _callBacks,

            verbose = verbose
        )

        return history

    def loadFromFile(self, modelFile):
        self.model.load_weights(modelFile)

    def saveModelToFile(self, modelFile):
        self.model.save_weights(modelFile)







@timeit
def trainPeakBotModel(trainInstancesPath, logBaseDir, modelName = None, valInstancesPath = None, addValidationInstances = None, everyNthEpoch = -1, verbose = False):
    tic("pbTrainNewModel")

    ## name new model
    if modelName is None:
        modelName = "%s%s__%s"%(Config.NAME, Config.VERSION, uuid.uuid4().hex[0:6])

    ## log information to folder
    logDir = os.path.join(logBaseDir, modelName)
    logger = tf.keras.callbacks.CSVLogger(os.path.join(logDir, "clog.tsv"), separator="\t")

    if verbose:
        print("Training new PeakBot model")
        print("  | Model name is '%s'"%(modelName))
        print("  | .. config is")
        print(Config.getAsStringFancy().replace(";", "\n"))
        print("  |")

    ## define learning rate schedule
    def lrSchedule(epoch, lr):
        if (epoch + 1) % Config.LEARNINGRATEDECREASEAFTERSTEPS == 0:
            lr *= Config.LEARNINGRATEMULTIPLIER
        tf.summary.scalar('learningRate', data=lr, step=epoch)

        return max(lr, Config.LEARNINGRATEMINVALUE)
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(lrSchedule, verbose=False)

    ## create generators for training data (and validation data if available)
    datGenTrain = modelAdapterTrainGenerator(dataGenerator(trainInstancesPath, verbose = verbose), newBatchSize = Config.BATCHSIZE, verbose = verbose)
    datGenVal   = None
    if valInstancesPath is not None:
        datGenVal = modelAdapterTrainGenerator(dataGenerator(valInstancesPath  , verbose = verbose), newBatchSize = Config.BATCHSIZE, verbose = verbose)
        datGenVal = tf.data.Dataset.from_tensors(next(datGenVal))

    ## add additional validation datasets to monitor model performance during the trainging process
    valDS = AdditionalValidationSets(logDir,
                                     batch_size=Config.BATCHSIZE,
                                     everyNthEpoch = everyNthEpoch,
                                     verbose = verbose)
    if addValidationInstances is not None:
        if verbose:
            print("  | Additional validation datasets")
        for valInstance in addValidationInstances:
            x,y = None, None
            if "folder" in valInstance.keys():
                print("  | .. - adding", valInstance["folder"])
                datGen  = modelAdapterTrainGenerator(dataGenerator(valInstance["folder"]),
                                                     newBatchSize = Config.BATCHSIZE)
                numBatches = valInstance["numBatches"] if "numBatches" in valInstance.keys() else 1
                x,y = convertGeneratorToPlain(datGen, numBatches)
            if "x" in valInstance.keys():
                x = valInstance["x"]
                y = valInstance["y"]
            if x is not None and y is not None and "name" in valInstance.keys():
                valDS.addValidationSet((x,y, valInstance["name"]))
                if verbose:
                    print("  | .. %s: %d instances"%(valInstance["name"], x["channel.int"].shape[0]))

            else:
                raise RuntimeError("Unknonw additional validation dataset")
        if verbose:
            print("  |")

    ## instanciate a new model and set its parameters
    pb = PeakBot(modelName)
    pb.buildTFModel(mode="training", verbose = verbose)

    ## train the model
    history = pb.train(
        datTrain = datGenTrain,
        datVal   = datGenVal,

        logDir = logDir,
        callbacks = [logger, lrScheduler, valDS],

        verbose = verbose * 2
    )


    ## save metrices of the training process in a user-convenient format (pandas table)
    metricesAddValDS = pd.DataFrame(columns=["model", "set", "metric", "value"])
    if addValidationInstances is not None:
        hist = valDS.history[-1]
        for valInstance in addValidationInstances:
            se = valInstance["name"]
            for metric in ["loss", "pred.peak_loss", "pred_loss", "pred.peak_categorical_accuracy", "pred.peak_MatthewsCorrelationCoefficient", "pred.rtInds_MSE", "pred_EICIOU", "pred_EICIOUPeaks"]:
                val = hist[se + "_" + metric]
                newRow = pd.Series({"model": modelName, "set": se, "metric": metric, "value": val})
                metricesAddValDS = metricesAddValDS.append(newRow, ignore_index=True)

    if verbose:
        print("  |")
        print("  | .. model built and trained successfully (took %.1f seconds)"%toc("pbTrainNewModel"))

    return pb, metricesAddValDS














def loadModel(modelPath, mode, verbose = True):
    pb = PeakBot("")
    pb.buildTFModel(mode=mode, verbose = verbose)
    pb.loadFromFile(modelPath)
    return pb






@timeit
def runPeakBot(instances, modelPath = None, model = None, verbose = True):
    tic("detecting with peakbot")

    if verbose:
        print("Detecting peaks with PeakBot")
        print("  | .. loading PeakBot model '%s'"%(modelPath))

    pb = model
    if model is None and modelPath is not None:
        model = loadModel(modelPath, mode = "predict")

    assert all(np.amax(instances["channel.int"], (1)) <= 1), "channel.int is not scaled to a maximum of 1 '%s'"%(str(np.amax(instances["channel.int"], (1))))
    
    pred = pb.model.predict(instances["channel.int"], verbose = verbose)
    peakTypes = pred[0]
    rtStartInds = pred[1][:,0]
    rtEndInds = pred[1][:,1]

    return peakTypes, rtStartInds, rtEndInds



@timeit
def integrateArea(eic, rts, start, end, method = "linear"):
    startInd = np.argmin([abs(r-start) for r in rts])
    endInd = np.argmin([abs(r-end) for r in rts])

    area = 0
    if method == "linear":
        for i in range(startInd, endInd+1):
            area = area + eic[i] - (rts[i]-rts[startInd])/(rts[endInd]-rts[startInd]) * (max(eic[startInd], eic[endInd]) - min(eic[startInd], eic[endInd])) + min(eic[startInd], eic[endInd])
    
    return area




@timeit
def evaluatePeakBot(instancesWithGT, modelPath = None, model = None, verbose = True):
    tic("detecting with peakbot")

    if verbose:
        print("Evaluating peaks with PeakBot")
        print("  | .. loading PeakBot model '%s'"%(modelPath))

    pb = model
    if model is None and modelPath is not None:
        pb = loadModel(modelPath, mode = "training")

    assert all(np.amax(instancesWithGT["channel.int"], (1)) <= 1), "channel.int is not scaled to a maximum of 1 '%s'"%(str(np.amax(instancesWithGT["channel.int"], (1))))
    
    x = {"channel.int": instancesWithGT["channel.int"]}
    y = {"pred"  : np.hstack((instancesWithGT["inte.peak"], instancesWithGT["inte.rtInds"], instancesWithGT["channel.int"])), "pred.peak": instancesWithGT["inte.peak"], "pred.rtInds": instancesWithGT["inte.rtInds"]}
    history = pb.model.evaluate(x, y, return_dict = True, verbose = verbose)

    return history

























def importTargets(targetFile, excludeSubstances = None, includeSubstances = None):
    if excludeSubstances is None:
        excludeSubstances = []

    ## load targets
    print("Loading targets from file '%s'"%(targetFile))
    headers, substances = readTSVFile(targetFile, header = True, delimiter = "\t", convertToMinIfPossible = True, getRowsAsDicts = True)
    substances = dict((substance["Name"], 
                       {"Name"     : substance["Name"].replace(" (ISTD)", ""),
                        "Q1"       : substance["Precursor Ion"],
                        "Q3"       : substance["Product Ion"],
                        "RT"       : substance["RT"],
                        "PeakForm" : substance["PeakForm"], 
                        "Rt shifts": substance["RT shifts"],
                        "Note"     : substance["Note"],
                        "Pola"     : substance["Ion Polarity"],
                        "ColE"     : None}) for substance in substances if substance["Name"] not in excludeSubstances and (includeSubstances is None or substance["Name"] in includeSubstances))
                ##TODO include collisionEnergy here
    print("  | .. loaded %d substances"%(len(substances)))
    print("  | .. of these %d have RT shifts"%(sum((1 if substance["Rt shifts"]!="" else 0 for substance in substances.values()))))
    print("  | .. of these %d have abnormal peak forms"%(sum((1 if substance["PeakForm"]!="" else 0 for substance in substances.values()))))
    print("\n")
    # targets: [{'Name': 'Valine', 'Q1': 176.0, 'Q3': 116.0, 'RT': 1.427}, ...]

    return substances


def loadIntegrations(substances, curatedPeaks):
    ## load integrations
    print("Loading integrations from file '%s'"%(curatedPeaks))
    headers, temp = parseTSVMultiLineHeader(curatedPeaks, headerRowCount=2, delimiter = ",", commentChar = "#", headerCombineChar = "$")
    headers = dict((k.replace(" (ISTD)", ""), v) for k,v in headers.items())
    foo = set([head[:head.find("$")] for head in headers if not head.startswith("Sample$")])
    notUsingSubs = []
    for substance in substances.values():
        if substance["Name"] not in foo:
            notUsingSubs.append(substance["Name"])
    if len(notUsingSubs) > 0:
        print("  | .. Not using %d substances (%s) as these are not in the integration matrix"%(len(notUsingSubs), ", ".join(notUsingSubs)))
    
    foo = dict((k, v) for k, v in substances.items() if k in foo)
    print("  | .. restricting substances from %d to %d (overlap of substances and integration results)"%(len(substances), len(foo)))
    substances = foo

    ## process integrations
    integrations = {}
    integratedSamples = set()
    totalIntegrations = 0
    foundPeaks = 0
    foundNoPeaks = 0
    for substance in [substance["Name"] for substance in substances.values()]:
        integrations[substance] = {}
        for intei, inte in enumerate(temp):
            area = inte[headers["%s$Area"%(substance)]]
            if area == "" or float(area) == 0:
                integrations[substance][inte[headers["Sample$Name"]]] = {"foundPeak": False,
                                                                        "rtstart"  : -1, 
                                                                        "rtend"    : -1, 
                                                                        "area"     : -1,
                                                                        "chrom"    : [],}
                foundNoPeaks += 1
            else:
                integrations[substance][inte[headers["Sample$Name"]]] = {"foundPeak": True,
                                                                        "rtstart"  : float(inte[headers["%s$Int. Start"%(substance)]]), 
                                                                        "rtend"    : float(inte[headers["%s$Int. End"  %(substance)]]), 
                                                                        "area"     : float(inte[headers["%s$Area"      %(substance)]]),
                                                                        "chrom"    : [],}
                foundPeaks += 1
            integratedSamples.add(inte[headers["Sample$Name"]])
            totalIntegrations += 1
    print("  | .. parsed %d integrations from %d substances and %d samples."%(totalIntegrations, len(substances), len(integratedSamples)))
    print("  | .. there are %d areas and %d no peaks"%(foundPeaks, foundNoPeaks))
    print("\n")
    # integrations [['Pyridinedicarboxylic acid Results', 'R100140_METAB02_MCC025_CAL1_20200306', '14.731', '14.731', '0'], ...]

    return substances, integrations


 

def loadChromatogramsTo(substances, integrations, samplesPath, expDir, loadFromPickleIfPossible = True,
                        allowedMZOffset = 0.05, MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)"):
    ## load chromatograms
    tic("procChroms")
    print("Processing chromatograms")
    samples = [os.path.join(samplesPath, f) for f in os.listdir(samplesPath) if os.path.isfile(os.path.join(samplesPath, f)) and f.lower().endswith(".mzml")]
    usedSamples = set()
    if os.path.isfile(os.path.join(expDir, "integrations.pickle")) and loadFromPickleIfPossible:
        with open(os.path.join(expDir, "integrations.pickle"), "rb") as fin:
            integrations, referencePeaks, noReferencePeaks, usedSamples = pickle.load(fin)
            print("  | .. Imported integrations from pickle file '%s/integrations.pickle'"%(expDir))
    else:
        print("  | .. This might take a couple of minutes as all samples/integrations/channels/etc. need to be compared and the current implementation are 4 sub-for-loops")
        for sample in tqdm.tqdm(samples):
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

                    collisionType = None
                    if entry.get_element_by_name("collision-induced dissociation") is not None:
                        collisionType = "collision-induced dissociation"

                    chrom = [(time, intensity) for time, intensity in entry.peaks()]

                    allChannels.append([Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entry.ID, chrom])

            ## merge channels with integration results for this sample
            for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID, chrom) in enumerate(allChannels):
                usedChannel = []
                useChannel = True
                ## test if channel is unique ## TODO include collisionEnergy here as well
                for bi, (bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionType, bentryID, bchrom) in enumerate(allChannels):
                    if i != bi:
                        if abs(Q1 - bq1) <= allowedMZOffset and abs(Q3 - bq3) <= allowedMZOffset and \
                            polarity == bpolarity and collisionType == bcollisionType:# TODO include collisionEnergy test here and collisionEnergy == bcollisionEnergy:
                            useChannel = False
                            unusedChannels.append(entryID)
                
                ## use channel if it is unique and find the integrated substance(s) for it
                if useChannel:
                    for substance in substances.values(): ## TODO include collisionEnergy check here
                        if abs(substance["Q1"] - Q1) < allowedMZOffset and abs(substance["Q3"] - Q3) <= allowedMZOffset and rtstart <= substance["RT"] <= rtend:
                            if substance["Name"] in integrations.keys() and sampleName in integrations[substance["Name"]].keys():
                                foundTargets.append([substance, entry, integrations[substance["Name"]][sampleName]])
                                usedChannel.append(substance)
                                integrations[substance["Name"]][sampleName]["chrom"].append(["%s (%s mode, %s with %.1f energy)"%(entryID, polarity, collisionType, collisionEnergy), 
                                                                                            Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID, chrom])
        
        ## remove all integrations with more than one scanEvent
        referencePeaks = 0
        noReferencePeaks = 0
        for substance in integrations.keys():
            for sample in integrations[substance].keys():
                if len(integrations[substance][sample]["chrom"]) == 1:
                    referencePeaks += 1
                else:
                    noReferencePeaks += 1
                    integrations[substance][sample]["chrom"].clear()

        with open (os.path.join(expDir, "integrations.pickle"), "wb") as fout:
            pickle.dump((integrations, referencePeaks, noReferencePeaks, usedSamples), fout)
            print("  | .. Stored integrations to '%s/integrations.pickle'"%expDir)
        
    print("  | .. There are %d peaks and %d no peaks"%(referencePeaks, noReferencePeaks))
    print("  | .. Using %d samples "%(len(usedSamples)))
    remSubstancesChannelProblems = []
    for substance in integrations.keys():
        foundOnce = False
        for sample in integrations[substance].keys():
            if len(integrations[substance][sample]["chrom"]) > 1:
                remSubstancesChannelProblems.append(substance)
                break
            elif len(integrations[substance][sample]["chrom"]) == 1:
                foundOnce = True
        if not foundOnce:
            remSubstancesChannelProblems.append(substance)
    if len(remSubstancesChannelProblems):
        print("  | .. %d substances (%s) were not found as the channel selection was ambiguous"%(len(remSubstancesChannelProblems), ", ".join(sorted(remSubstancesChannelProblems))))
        print("  | .. These will not be used further")
        for r in remSubstancesChannelProblems:
            del integrations[r]
    print("  | .. took %.1f seconds"%(toc("procChroms")))
    print("\n")

    return substances, integrations

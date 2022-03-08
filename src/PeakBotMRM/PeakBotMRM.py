import logging
from tkinter import N
import traceback
import warnings

from .core import *

import sys
import os
import pickle
import uuid
import re
import math

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

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
    VERSION = "0.9"

    RTSLICES       = 256   ## should be of 2^n
    NUMCLASSES     =   2   ## [chromatographicPeak, noPeak]

    BATCHSIZE      =  16
    STEPSPEREPOCH  =   8
    EPOCHS         = 300

    DROPOUT        = 0.2
    UNETLAYERSIZES = [32,64,128,256]

    LEARNINGRATESTART              = 0.005
    LEARNINGRATEDECREASEAFTERSTEPS = 5
    LEARNINGRATEMULTIPLIER         = 0.96
    LEARNINGRATEMINVALUE           = 3e-17

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



print("Initializing PeakBotMRM")
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
    pus = tf.config.experimental.list_physical_devices()
    for pu in pus:
        print("  | .. TensorFlow device: Name '%s', type '%s'"%(pu.name, pu.device_type))
except Exception:
    print("  | .. fetching GPU info failed")
print("")
    










#####################################
### Data generator methods
### Read files from a directory and prepare them
### for PeakBotMRM training and prediction
##
def getSizeOfTraningSet(folder, instancePrefix = None, batchSize = None, verbose=False):
    if instancePrefix is None:
        instancePrefix = Config.INSTANCEPREFIX
    if batchSize is None:
        batchSize = Config.BATCHSIZE
        
    ite = 0
    total = 0
    while os.path.isfile(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite))):
        l = pickle.load(open(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite)), "rb"))
        
        k = l[[k for k in l.keys()][0]]
        bs = k.shape[0] if isinstance(k, np.ndarray) else len(k)
        
        ite += 1
        total += bs
    return math.floor(total/batchSize)

def dataGenerator(folder, instancePrefix = None, batchSize = None, verbose=False):
    if instancePrefix is None:
        instancePrefix = Config.INSTANCEPREFIX
    if batchSize is None:
        batchSize = Config.BATCHSIZE

    ite = 0
    l = None
    while os.path.isfile(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite))):
        lc = pickle.load(open(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite)), "rb"))
        if l is None:
            l = lc
        else:
            for k in lc.keys():
                if isinstance(l[k], np.ndarray):
                    if len(l[k].shape)==1:
                        l[k] = np.concatenate((l[k], lc[k]))
                    else:
                        l[k] = np.vstack((l[k], lc[k]))

                elif isinstance(lc[k], list):
                    l[k] = l[k] + lc[k]
        
        k = l[[k for k in l.keys()][0]]
        bs = k.shape[0] if isinstance(k, np.ndarray) else len(k)
        
        while batchSize <= bs and bs > 0:
            if batchSize > 0:
                temp = {}
                for k in list(l.keys()):
                    if   isinstance(l[k], np.ndarray) and len(l[k].shape)==1:
                        temp[k] = l[k][0:batchSize]
                        
                    if   isinstance(l[k], np.ndarray) and len(l[k].shape)==2:
                        temp[k] = l[k][0:batchSize,:]
                        l[k]

                    elif isinstance(l[k], np.ndarray) and len(l[k].shape)==3:
                        temp[k] = l[k][0:batchSize,:,:]

                    elif isinstance(l[k], np.ndarray) and len(l[k].shape)==4:
                        temp[k] = l[k][0:batchSize,:,:,:]

                    elif isinstance(l[k], list):
                        temp[k] = l[k][0:batchSize]
                    
                    l[k] = l[k][batchSize:]
                    if len(l[k]) == 0:
                        del l[k]
                
                if len(l) == 0:
                    l = None 
                    bs = 0
                else:
                    k = l[[k for k in l.keys()][0]]
                    bs = k.shape[0] if isinstance(k, np.ndarray) else len(k)
                
            else:
                temp = l
                bs = 0
                l = None
            
            yield temp
            temp = None
            
        ite += 1

def modelAdapterGenerator(datGen, xKeys, yKeys, verbose=False):
    for l in datGen:
        if "channel.int" in l.keys() and "inte.peak" in l.keys() and "inte.rtInds" in l.keys():
            l["pred"] = np.hstack((l["inte.peak"], l["inte.rtInds"], l["channel.int"]))
        x = dict((xKeys[k],v) for k,v in l.items() if k in xKeys.keys())
        y = dict((yKeys[k],v) for k,v in l.items() if k in yKeys.keys())

        yield x,y

def modelAdapterTrainGenerator(datGen, verbose=False):
    temp = modelAdapterGenerator(datGen, 
                                 {"channel.int":"channel.int"}, 
                                 {"pred": "pred", "inte.peak":"pred.peak", "inte.rtInds": "pred.rtInds"},
                                 verbose = verbose)
    return temp

def modelAdapterPredictGenerator(datGen, verbose=False):
    temp = modelAdapterGenerator(datGen, 
                                 {"channel.int":"channel.int"}, 
                                 {}, 
                                 verbose = verbose)
    return temp


def convertGeneratorToPlain(gen):
    x = None
    y = None
    for t in gen:
        if x is None:
            x = t[0]
            y = t[1]
        else:
            for k in x.keys():
                if isinstance(x[k], list):
                    x[k].append(t[0][k])
                elif isinstance(x[k], np.ndarray):
                    x[k] = np.concatenate((x[k], t[0][k]), axis=0)
                else:
                    assert False, "Unknown key, aborting"
            for k in y.keys():
                if isinstance(y[k], list):
                    y[k].append(t[1][k])
                elif isinstance(y[k], np.ndarray):
                    y[k] = np.concatenate((y[k], t[1][k]), axis=0)
                else:
                    assert False, "Unknown key, aborting"
                        
    return x, y



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
            print("  | .. Pre-processing")
            print("  | .. .. each eic is baseline corrected (signal with minimum abundance) and afterwards scaled to 1 (signal with maximum abundance)")
            print("  | ")
        
        ## Normalize and scale EIC (remove constant baseline and scale to a maximum intensity value of 1)
        minVal = eicValMin
        minVal = tf.repeat(minVal, repeats=[Config.RTSLICES], axis=1)
        eic = tf.math.subtract(eic, minVal)

        maxVal = tf.where(eicValMax == 0, tf.ones_like(eicValMax), eicValMax)
        maxVal = tf.repeat(maxVal, repeats=[Config.RTSLICES], axis=1)
        eic = tf.math.divide(eic, maxVal)
        
        if verbose:            
            print("  | .. Preprocessing")
            print("  | .. .. normalization and scaling (for each standardized EIC: 1) subtraction of minimum value; 2) division by maximum value")
            print("  |")

        ## add "virtual" channel to the EICs as required by the convolutions
        eic = tf.expand_dims(eic, axis=-1)

        ## setup convolutions
        x = eic
        for i in range(len(uNetLayerSizes)):
            if True:
                
                x = tf.keras.layers.ZeroPadding1D(padding=2)(x)
                x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (5), use_bias=False, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)

                x = tf.keras.layers.ZeroPadding1D(padding=1)(x)
                x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (3), use_bias=False, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)

                x = tf.keras.layers.MaxPooling1D((2))(x)
                x = tf.keras.layers.Dropout(dropOutRate)(x)
            else:
                width = 7
                x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (2*width+1), padding="same", use_bias=False, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                
                width = 2
                x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (2*width+1), padding="same", use_bias=False, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                
                width = 1
                x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (2*width+1), padding="same", use_bias=False, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)

                x = tf.keras.layers.MaxPooling1D((2))(x)
                x = tf.keras.layers.Dropout(dropOutRate)(x)

        ## Intermediate layer
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        
        ## Predictions
        fx = x        
        peaks  = tf.keras.layers.Dense(Config.NUMCLASSES, activation="sigmoid", name="pred.peak")(fx)
        outputs.append(peaks)
        
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
            "pred": 1/1000 }
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
def trainPeakBotMRMModel(trainInstancesPath, logBaseDir, modelName = None, valInstancesPath = None, addValidationInstances = None, everyNthEpoch = -1, verbose = False):
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
        if (epoch + 1) % Config.LEARNINGRATEDECREASEAFTERSTEPS == 0:
            lr *= Config.LEARNINGRATEMULTIPLIER
        tf.summary.scalar('learningRate', data=lr, step=epoch)

        return max(lr, Config.LEARNINGRATEMINVALUE)
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(lrSchedule, verbose=False)

    ## create generators for training data (and validation data if available)
    datGenTrain = modelAdapterTrainGenerator(dataGenerator(trainInstancesPath, verbose = verbose), verbose = verbose)
    epochs = getSizeOfTraningSet(trainInstancesPath)
    if verbose:
        print("  | .. There are %d training batches available"%(epochs))
        print("  |")
    
    datGenVal   = None
    if valInstancesPath is not None:
        datGenVal = modelAdapterTrainGenerator(dataGenerator(valInstancesPath, verbose = verbose), verbose = verbose)
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
            tic("addDS")
            x,y = None, None
            if "folder" in valInstance.keys():
                print("  | .. - adding", valInstance["folder"])
                datGen  = modelAdapterTrainGenerator(dataGenerator(valInstance["folder"], batchSize = -1))
                x, y = convertGeneratorToPlain(datGen)
                
            if "x" in valInstance.keys():
                x = valInstance["x"]
                y = valInstance["y"]
                
            if x is not None and y is not None and "name" in valInstance.keys():
                valDS.addValidationSet((x,y, valInstance["name"]))
                if verbose:
                    print("  | .. .. %s: %d instances (adding took %.1f sec)"%(valInstance["name"], x["channel.int"].shape[0], toc("addDS")))

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
    metricesAddValDS = pd.DataFrame(columns=["model", "set", "metric", "value"])
    if addValidationInstances is not None:
        hist = valDS.history[-1]
        for valInstance in addValidationInstances:
            se = valInstance["name"]
            for metric, metName in {"loss":"loss", "pred.peak_MatthewsCorrelationCoefficient":"MCC", "pred_EICIOUPeaks":"Area IOU", "pred.peak_Acc4Peaks": "Sensitivity (peaks)", "pred.peak_Acc4NonPeaks": "Specificity (no peaks)", "pred.peak_categorical_accuracy": "Accuracy"}.items():
                val = hist[se + "_" + metric]
                newRow = pd.Series({"model": modelName, "set": se, "metric": metName, "value": val})
                metricesAddValDS = metricesAddValDS.append(newRow, ignore_index=True)

    if verbose:
        print("  |")
        print("  | .. model built and trained successfully (took %.1f seconds)"%toc("pbTrainNewModel"))

    return pb, metricesAddValDS, modelName



@timeit
def integrateArea(eic, rts, start, end, method = "minbetweenborders"):
    startInd = np.argmin([abs(r-start) for r in rts])
    endInd = np.argmin([abs(r-end) for r in rts])

    if end <= start:
        warnings.warn("Warning in peak area calculation: start and end rt of peak are incorrect (start %.2f, end %.2f). An area of 0 will be returned."%(start, end), RuntimeWarning)
        return 0


    area = 0
    if method.lower() in ["linearbetweenborders", "linear"]:
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
        minV = None
        for i in range(startInd, endInd+1):
            if minV is None or minV > eic[i]:
                minV = eic[i]
            area = area + eic[i]
        if minV is None:
            minV = 0
        area = area - minV * (endInd - startInd + 1)

    else:
        raise RuntimeError("Unknown integration method")

    
    return area














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

    pb = model
    if model is None and modelPath is not None:
        model = loadModel(modelPath, mode = "predict")
    
    pred = pb.model.predict(instances["channel.int"], verbose = verbose)
    peakTypes = np.argmax(pred[0], axis=1)
    rtStartInds = pred[1][:,0]
    rtEndInds = pred[1][:,1]

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

























def loadTargets(targetFile, excludeSubstances = None, includeSubstances = None, verbose = True, logPrefix = ""):
    if excludeSubstances is None:
        excludeSubstances = []

    ## load targets
    if verbose: 
        print(logPrefix, "Loading targets from file '%s'"%(targetFile))
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
                        "CE"       : None,
                        "CET"      : None}) for substance in substances if substance["Name"] not in excludeSubstances and (includeSubstances is None or substance["Name"] in includeSubstances))
                ##TODO include collisionEnergy here
    if verbose:
        print(logPrefix, "  | .. loaded %d substances"%(len(substances)))
        print(logPrefix, "  | .. of these %d have RT shifts"%(sum((1 if substance["Rt shifts"]!="" else 0 for substance in substances.values()))))
        print(logPrefix, "  | .. of these %d have abnormal peak forms"%(sum((1 if substance["PeakForm"]!="" else 0 for substance in substances.values()))))
        print(logPrefix, "\n", logPrefix)

    return substances



def loadIntegrations(substances, curatedPeaks, verbose = True, logPrefix = ""):
    ## load integrations
    if verbose:
        print(logPrefix, "Loading integrations from file '%s'"%(curatedPeaks))
    headers, integrationData = parseTSVMultiLineHeader(curatedPeaks, headerRowCount=2, delimiter = ",", commentChar = "#", headerCombineChar = "$")
    headers = dict((k.replace(" (ISTD)", ""), v) for k,v in headers.items())
    foo = set([header[:header.find("$")] for header in headers if not header.startswith("Sample$")])
    notUsingSubstances = []
    for substance in substances.values():
        if substance["Name"] not in foo:
            notUsingSubstances.append(substance["Name"])
    if verbose and len(notUsingSubstances) > 0:
        print(logPrefix, "  | .. Not using %d substances (%s) as these are not in the integration matrix"%(len(notUsingSubstances), ", ".join(notUsingSubstances)))
    
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
    for substance in [substance["Name"] for substance in substances.values()]:
        integrations[substance] = {}
        for i, integration in enumerate(integrationData):
            area = integration[headers["%s$Area"%(substance)]]
            if area == "" or float(area) == 0:
                integrations[substance][integration[headers["Sample$Name"]]] = {"foundPeak": False,
                                                                        "rtstart"  : -1, 
                                                                        "rtend"    : -1, 
                                                                        "area"     : -1,
                                                                        "chrom"    : [],}
                foundNoPeaks += 1
            else:
                integrations[substance][integration[headers["Sample$Name"]]] = {"foundPeak": True,
                                                                        "rtstart"  : float(integration[headers["%s$Int. Start"%(substance)]]), 
                                                                        "rtend"    : float(integration[headers["%s$Int. End"  %(substance)]]), 
                                                                        "area"     : float(integration[headers["%s$Area"      %(substance)]]),
                                                                        "chrom"    : [],}
                foundPeaks += 1
            integratedSamples.add(integration[headers["Sample$Name"]])
            totalIntegrations += 1
    if verbose:
        print(logPrefix, "  | .. parsed %d integrations from %d substances and %d samples."%(totalIntegrations, len(substances), len(integratedSamples)))
        print(logPrefix, "  | .. there are %d areas and %d no peaks"%(foundPeaks, foundNoPeaks))
        print(logPrefix, "\n", logPrefix)
    # integrations [['Pyridinedicarboxylic acid Results', 'R100140_METAB02_MCC025_CAL1_20200306', '14.731', '14.731', '0'], ...]

    return substances, integrations


 
def loadChromatograms(substances, integrations, samplesPath, expDir, loadFromPickleIfPossible = True,
                        allowedMZOffset = 0.05, MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                        verbose = True, logPrefix = ""):
    ## load chromatograms
    tic("procChroms")
    if verbose:
        print(logPrefix, "Processing chromatograms")
    samples = [os.path.join(samplesPath, f) for f in os.listdir(samplesPath) if os.path.isfile(os.path.join(samplesPath, f)) and f.lower().endswith(".mzml")]
    usedSamples = set()
    if os.path.isfile(os.path.join(expDir, "integrations.pickle")) and loadFromPickleIfPossible:
        with open(os.path.join(expDir, "integrations.pickle"), "rb") as fin:
            integrations, usedSamples = pickle.load(fin)
            if verbose:
                print(logPrefix, "  | .. Imported integrations from pickle file '%s/integrations.pickle'"%(expDir))
    else:
        if verbose:
            print(logPrefix, "  | .. This might take a couple of minutes as all samples/integrations/channels/etc. need to be compared and the current implementation are 4 sub-for-loops")
        for sample in tqdm.tqdm(samples, desc="  | .. importing"):
            sampleName = os.path.basename(sample)
            if verbose: 
                print("logPrefix, \nsample", sampleName)
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

                    collisionType = None
                    if entry.get_element_by_name("collision-induced dissociation") is not None:
                        collisionType = "collision-induced dissociation"
                    if collisionType == "collision-induced dissociation":
                        collisionType = "CID"

                    rts = np.array([time for time, intensity in entry.peaks()])
                    eic = np.array([intensity for time, intensity in entry.peaks()])
                    chrom = {"rts": rts, "eic": eic}

                    allChannels.append([Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entry.ID, chrom])

            ## merge channels with integration results for this sample
            for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID, chrom) in enumerate(allChannels):
                usedChannel = []
                useChannel = True
                ## test if channel is unique
                for bi, (bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionType, bentryID, bchrom) in enumerate(allChannels):
                    if i != bi:  ## correct, cannot be optimized as both channels (earlier and later) shall not be used in case of a collision
                        if abs(Q1 - bq1) <= allowedMZOffset and abs(Q3 - bq3) <= allowedMZOffset and \
                            ((rtstart <= brtstart <= rtend) or (rtstart <= brtend <= rtend)) and \
                            polarity == bpolarity and \
                            collisionType == bcollisionType: # TODO include collisionEnergy test here:
                            useChannel = False
                            unusedChannels.append(entryID)
                            if verbose: 
                                print(logPrefix, "Problematic channel combination found. Both will be skipped (TODO implement CE reference from the reference data)")
                                print(logPrefix, "  channel     Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, pol '%10s', CE %5.1f, Method '%s', Header '%s'"%(Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID))
                                print(logPrefix, "  problematic Q1 %8.3f, Q3 %8.3f, Rt %5.2f - %5.2f, pol '%10s', CE %5.1f, Method '%s', Header '%s'"%(bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionType, bentryID))
                                print(logPrefix, )
                
                ## use channel if it is unique and find the integrated substance(s) for it
                if useChannel:
                    for substance in substances.values(): ## TODO include collisionEnergy and collisionMethod checks here
                        if abs(substance["Q1"] - Q1) < allowedMZOffset and abs(substance["Q3"] - Q3) <= allowedMZOffset and \
                            rtstart <= substance["RT"] <= rtend:
                            if substance["Name"] in integrations.keys() and sampleName in integrations[substance["Name"]].keys():
                                foundTargets.append([substance, entry, integrations[substance["Name"]][sampleName]])
                                usedChannel.append(substance)
                                integrations[substance["Name"]][sampleName]["chrom"].append(["%s (%s mode, %s with %.1f CE)"%(entryID, polarity, collisionType, collisionEnergy), 
                                                                                            Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID, chrom])
        

        with open (os.path.join(expDir, "integrations.pickle"), "wb") as fout:
            pickle.dump((integrations, usedSamples), fout)
            if verbose:
                print(logPrefix, "  | .. Stored integrations to '%s/integrations.pickle'"%expDir)
    
    ## Remove chromatograms with ambiguously selected chromatograms
    remSubstancesChannelProblems = set()
    for substance in integrations.keys():
        foundOnce = False
        for sample in integrations[substance].keys():
            if len(integrations[substance][sample]["chrom"]) == 1:
                foundOnce = True
            elif len(integrations[substance][sample]["chrom"]) != 1:
                remSubstancesChannelProblems.add(substance)
                break
        if not foundOnce:
            remSubstancesChannelProblems.add(substance)
    if len(remSubstancesChannelProblems):
        if verbose:
            print(logPrefix, "  | .. %d substances (%s) were not found as the channel selection was ambiguous. These will not be used further"%(len(remSubstancesChannelProblems), ", ".join(sorted(remSubstancesChannelProblems))))
            print(logPrefix, "  | .. ATTENTION: CE and CET are not yet available (as the reference does not specify these values). Thus, some of the reference compounds cannot be used due to ambiguous channel selection")
        for r in remSubstancesChannelProblems:
            del integrations[r]
    
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
    if verbose:
        print(logPrefix, "  | .. There are %d sample.substances with unambiguous chromatograms (and %d sample.substances with ambiguous chromatograms) from %d samples"%(referencePeaks, noReferencePeaks, len(usedSamples)))
    
    if verbose:
        print(logPrefix, "  | .. took %.1f seconds"%(toc("procChroms")))
        print(logPrefix, "\n", logPrefix)

    return substances, integrations

import logging

from .core import tic, toc, tocAddStat, timeit, writeTSVFile

import os
import pickle
import uuid

import tensorflow as tf
import numpy as np
import pandas as pd




#####################################
### Configuration class
##
class Config(object):
    """Base configuration class"""

    NAME    = "PeakBot"
    VERSION = "0.9"

    RTSLICES       = 256   ## should be of 2^n
    NUMCLASSES     =   2   ## [isFullPeak, hasCoelutingPeakLeftAndRight, hasCoelutingPeakLeft, hasCoelutingPeakRight, isWall, isBackground]
    FIRSTNAREPEAKS =   1   ## specifies which of the first n classes represent a chromatographic peak (i.e. if classes 0,1,2,3 represent a peak, the value for this parameter must be 4)

    BATCHSIZE     =   16
    STEPSPEREPOCH =    8
    EPOCHS        =  110

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
            " Size of EIC: %d (scans)"%(Config.RTSLICES),
            " Number of peak-classes: %d"%(Config.NUMCLASSES),
            " Batchsize: %d, Epochs %d, StepsPerEpoch: %d"%(Config.BATCHSIZE, Config.EPOCHS, Config.STEPSPEREPOCH),
            " DropOutRate: %g"%(Config.DROPOUT),
            " UNetLayerSizes: %s"%(Config.UNETLAYERSIZES),
            " LearningRate: Start: %g, DecreaseAfter: %d steps, Multiplier: %g, min. rate: %g"%(Config.LEARNINGRATESTART, Config.LEARNINGRATEDECREASEAFTERSTEPS, Config.LEARNINGRATEMULTIPLIER, Config.LEARNINGRATEMINVALUE),
            " Prefix for instances: '%s'"%Config.INSTANCEPREFIX,
            " "
        ])

    @staticmethod
    def getAsString():
        return ";".join([
            "%s"%(Config.NAME),
            "Version " + Config.VERSION,
            "Size of EIC: %d (scans)"%(Config.RTSLICES),
            "Number of peak-classes: %d"%(Config.NUMCLASSES),
            "Batchsize: %d, Epochs %d, StepsPerEpoch: %d"%(Config.BATCHSIZE, Config.EPOCHS, Config.STEPSPEREPOCH),
            "DropOutRate: %g"%(Config.DROPOUT),
            "UNetLayerSizes: %s"%(Config.UNETLAYERSIZES),
            "LearningRate: Start: %g, DecreaseAfter: %d steps, Multiplier: %g, min. rate: %g"%(Config.LEARNINGRATESTART, Config.LEARNINGRATEDECREASEAFTERSTEPS, Config.LEARNINGRATEMULTIPLIER, Config.LEARNINGRATEMINVALUE),
            "InstancePrefix: '%s'"%(Config.INSTANCEPREFIX),
            ""
        ])



print("Initializing PeakBot")
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

        #print("Batch: number %d, %d peaks, %d backgrounds"%(ite, np.sum(l["inte.peak"][:,0]), np.sum(l["inte.peak"][:,1])))
        #print("channel.int", l["channel.int"])
        #print("inte.peak", l["inte.peak"])
        #print("inte.rtInds", l["inte.rtInds"])
        yield (dict((xKeys[k],v) for k,v in l.items() if k in xKeys.keys()), dict((yKeys[k],v) for k,v in l.items() if k in yKeys.keys()))
        l = next(datGen)
        ite += 1

def modelAdapterTrainGenerator(datGen, newBatchSize = None, verbose=False):
    temp = modelAdapterGenerator(datGen, 
                                 {"channel.int":"channel.int"},#, "inte.peak":"inte.peak", "inte.rtInds":"inte.rtInds"}, 
                                 {"inte.peak":"pred.peak", "inte.rtInds":"pred.rtInds"},#, "loss.IOU_Area":"loss.IOU_Area"}, 
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
def iou(boxes1, boxes2):
    ## from https://github.com/paperclip/cat-camera/blob/master/object_detection_2/core/post_processing.py
    """Calculates the overlap between proposal and ground truth boxes.
    Some `boxes2` may have been padded. The returned `iou` tensor for these
    boxes will be -1.
    Args:
    boxes1: a tensor with a shape of [batch_size, 2]. N is the number of
    proposals before groundtruth assignment. The last dimension is the pixel
    coordinates in [ymin, xmin, ymax, xmax] form.
    boxes2: a tensor with a shape of [batch_size, 2]. This
    tensor might have paddings with a negative value.
    Returns:
    iou: a tensor with as a shape of [batch_size].
    """
    with tf.name_scope('BatchIOU'):
        x1_min, x1_max = tf.split(value=boxes1, num_or_size_splits=2, axis=1)
        x2_min, x2_max = tf.split(value=boxes2, num_or_size_splits=2, axis=1)

        # Calculates the intersection area
        intersection_xmin = tf.maximum(x1_min, x2_min)
        intersection_xmax = tf.minimum(x1_max, x2_max)
        intersection_area = tf.maximum(tf.subtract(intersection_xmax, intersection_xmin), 0)

        # Calculates the union area
        area1 = tf.subtract(x1_max, x1_min)
        area2 = tf.subtract(x2_max, x2_min)
        # Adds a small epsilon to avoid divide-by-zero.
        union_area = tf.add(tf.subtract(tf.add(area1, area2), intersection_area), tf.constant(1e-8))

        # Calculates IoU
        iou = tf.divide(intersection_area, union_area)

        return iou
def InvIOU(boxes1, boxes2):
    return 1 - iou(boxes1, boxes2)

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def recall(y_true, y_pred):
    true_positives     = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall_keras

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def precision(y_true, y_pred):
    true_positives      = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision_keras

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def specificity(y_true, y_pred):
    tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + tf.keras.backend.epsilon())

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def negative_predictive_value(y_true, y_pred):
    tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + tf.keras.backend.epsilon())

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

def pF1(y_true, y_pred):
    return f1(tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1), Config.FIRSTNAREPEAKS), tf.float32),
              tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1), Config.FIRSTNAREPEAKS), tf.float32))
def pTPR(y_true, y_pred):
    return recall(tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1), Config.FIRSTNAREPEAKS), tf.float32),
                  tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1), Config.FIRSTNAREPEAKS), tf.float32))
def pFPR(y_true, y_pred):
    return 1-precision(tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1), Config.FIRSTNAREPEAKS), tf.float32),
                       tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1), Config.FIRSTNAREPEAKS), tf.float32))




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
                for i, (metric, result) in enumerate(zip(self.model.metrics_names, results)):
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
            print("  | PeakBot v %s model"%(self.version))
            print("  | .. Desc: Detection of a single LC-HRMS peak in an area")
            print("  | ")

        ## Input: Only LC-HRMS area
        input_ = tf.keras.Input(shape=(self.rts, 1), name="channel.int")
        inputs.append(input_)
        #if mode == "training":
        #    peaks_ = tf.keras.Input(shape=(Config.NUMCLASSES), name="inte.peak")
        #    inputs.append(peaks_)
        #    rtInds_ = tf.keras.Input(shape=(2), name="inte.rtInds")
        #    inputs.append(rtInds_)
        
        if verbose:
            print("  | .. Inputs")
            print("  | .. .. channel.int is", input_)
            #if mode == "training":
            #    print("  | .. .. inte.peak       is ", peaks_)
            #    print("  | .. .. inte.rtInds     is ", rtInds_)
            print("  |")

        ## Encoder
        x = input_
        cLayers = [x]
        for i in range(len(uNetLayerSizes)):

            x = tf.keras.layers.ZeroPadding1D(padding=2)(x)
            x = tf.keras.layers.Conv1D(uNetLayerSizes[i], (5), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

            #x = tf.keras.layers.Dropout(dropOutRate)(x)
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
        
        peaks  = tf.keras.layers.Dense(Config.NUMCLASSES, name="pred.peak", activation="sigmoid")(fx)
        outputs.append(peaks)
        rtInds = tf.keras.layers.Dense(2, activation="relu", name="pred.rtInds")(fx)
        outputs.append(rtInds)
        
        if False:
            overlap = None
            if mode == "training":
                #input_ = tf.keras.Input(shape=(256, 1), name="channel.int")
                #rtInds_ = tf.keras.Input(shape=(2), name="inte.rtInds")
                
                #input_ = tf.expand_dims(tf.convert_to_tensor([[i/2. for i in [0,0,0,1,2,1,1,1,0,0,0,1]], [i/9. for i in [0,0,0,0,2,4,9,6,3,1,0,0]], [i/11. for i in [0,0,2,4,11,6,3,1,0,0,0,0]]], dtype=tf.float32), axis=2)
                #rtInds_ = tf.convert_to_tensor([[0,0], [4,9], [2,7]], dtype=tf.float32)
                #rtInds = tf.convert_to_tensor([[0,8], [3,8], [2,7]], dtype=tf.float32)
                #peaks = tf.convert_to_tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]], dtype=tf.float32)

                indices = tf.cast(tf.transpose(tf.reshape(tf.repeat(tf.range(input_.shape[1]), repeats=[tf.shape(input_)[0]]), [1, input_.shape[1], tf.shape(input_)[0]])), dtype=input_.dtype)

                ## Extract area for user integration
                stripped = tf.where(tf.math.logical_and(indices >= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.floor(rtInds_[:,0]), dtype=input_.dtype), repeats=[input_.shape[1]]), [tf.shape(input_)[0], input_.shape[1]]), axis=2), 
                                                        indices <= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.ceil(rtInds_[:,1]), dtype=input_.dtype), repeats=[input_.shape[1]]), [tf.shape(input_)[0], input_.shape[1]]), axis=2)), 
                                    input_, tf.zeros_like(input_))
                maxRow = tf.where(stripped == 0, tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(input_.shape[1])), [tf.shape(input_)[0], input_.shape[1], input_.shape[2]]), stripped)
                minVal = tf.reduce_min(maxRow, axis=1)
                stripped = stripped - tf.reshape(tf.repeat(minVal, repeats=input_.shape[1]), [tf.shape(input_)[0], input_.shape[1], input_.shape[2]])
                stripped = tf.where(stripped < 0, tf.zeros_like(stripped), stripped)
                inteArea = tf.squeeze(tf.reduce_sum(stripped, axis=1), axis=1)

                ## Extract area for PeakBot_MRM integration
                stripped = tf.where(tf.math.logical_and(indices >= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.floor(rtInds[:,0]), dtype=input_.dtype), repeats=[input_.shape[1]]), [tf.shape(input_)[0], input_.shape[1]]), axis=2), 
                                                        indices <= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.ceil(rtInds[:,1]), dtype=input_.dtype), repeats=[input_.shape[1]]), [tf.shape(input_)[0], input_.shape[1]]), axis=2)), 
                                    input_, tf.zeros_like(input_))
                maxRow = tf.where(stripped == 0, tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(input_.shape[1])), [tf.shape(input_)[0], input_.shape[1], input_.shape[2]]), stripped)
                minVal = tf.reduce_min(maxRow, axis=1)
                stripped = stripped - tf.reshape(tf.repeat(minVal, repeats=input_.shape[1]), [tf.shape(input_)[0], input_.shape[1], input_.shape[2]])
                stripped = tf.where(stripped < 0, tf.zeros_like(stripped), stripped)
                pbCalcArea = tf.squeeze(tf.reduce_sum(stripped, axis=1), axis=1)

                ## Extract area for overlap of user and PeakBot_MRM integration
                beginInds = tf.math.maximum(rtInds_[:,0], tf.cast(tf.math.floor(rtInds[:,0]), dtype=rtInds_.dtype))
                endInds = tf.math.minimum(rtInds_[:,1], tf.cast(tf.math.ceil(rtInds[:,1]), dtype=rtInds_.dtype))
                stripped = tf.where(tf.math.logical_and(indices >= tf.expand_dims(tf.reshape(tf.repeat(beginInds, repeats=[input_.shape[1]]), [tf.shape(input_)[0], input_.shape[1]]), axis=2), 
                                                        indices <= tf.expand_dims(tf.reshape(tf.repeat(endInds, repeats=[input_.shape[1]]), [tf.shape(input_)[0], input_.shape[1]]), axis=2)), 
                                    input_, tf.zeros_like(input_))
                maxRow = tf.where(stripped == 0, tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(input_.shape[1])), tf.shape(input_)), stripped)
                minVal = tf.reduce_min(maxRow, axis=1)
                stripped = stripped - tf.reshape(tf.repeat(minVal, repeats=input_.shape[1]), [tf.shape(input_)[0], input_.shape[1], input_.shape[2]])
                stripped = tf.where(stripped < 0, tf.zeros_like(stripped), stripped)
                overlapArea = tf.squeeze(tf.reduce_sum(stripped, axis=1), axis=1)            
                #overlapArea = tf.where(tf.argmax(peaks, axis=1) == 1, tf.zeros_like(overlapArea), overlapArea)

                ## Calculate IOU
                overlap = tf.divide(overlapArea, tf.subtract(tf.add(inteArea, pbCalcArea), overlapArea))
                overlap = tf.keras.layers.Lambda(lambda x: x, name="loss.IOU_Area")(overlap)
                outputs.append(overlap)

        if verbose:
            print("  | .. Intermediate layer")
            print("  | .. .. lastUpLayer is", lastUpLayer)
            print("  | .. .. fx          is", fx)
            print("  | .. ")
            print("  | .. Outputs")
            print("  | .. .. pred.peak     is", peaks)
            print("  | .. .. pred.rtInds   is", rtInds)
            #if mode == "training":
            #    print("  | .. .. loss.IOU_Area is ", overlap)
            print("  | .. ")
            print("  | ")

        self.model = tf.keras.models.Model(inputs, outputs)
        
        if False:
            # TODO implement AreaIOU gradient for pred.rtInds
            def areaIOULoss(inteRTInds, predRTInds, channelInt):

                indices = tf.cast(tf.transpose(tf.reshape(tf.repeat(tf.range(channelInt.shape[1]), repeats=[tf.shape(channelInt)[0]]), [1, channelInt.shape[1], tf.shape(channelInt)[0]])), dtype=channelInt.dtype)

                ## Extract area for user integration
                stripped = tf.where(tf.math.logical_and(indices >= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.floor(inteRTInds[:,0]), dtype=channelInt.dtype), repeats=[channelInt.shape[1]]), [tf.shape(channelInt)[0], channelInt.shape[1]]), axis=2), 
                                                        indices <= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.ceil(inteRTInds[:,1]), dtype=channelInt.dtype), repeats=[channelInt.shape[1]]), [tf.shape(channelInt)[0], channelInt.shape[1]]), axis=2)), 
                                    channelInt, tf.zeros_like(channelInt))
                maxRow = tf.where(stripped == 0, tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(channelInt.shape[1])), [tf.shape(channelInt)[0], channelInt.shape[1], channelInt.shape[2]]), stripped)
                minVal = tf.reduce_min(maxRow, axis=1)
                stripped = stripped - tf.reshape(tf.repeat(minVal, repeats=channelInt.shape[1]), [tf.shape(channelInt)[0], channelInt.shape[1], channelInt.shape[2]])
                stripped = tf.where(stripped < 0, tf.zeros_like(stripped), stripped)
                inteArea = tf.squeeze(tf.reduce_sum(stripped, axis=1), axis=1)

                ## Extract area for PeakBot_MRM integration
                stripped = tf.where(tf.math.logical_and(indices >= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.floor(predRTInds[:,0]), dtype=channelInt.dtype), repeats=[channelInt.shape[1]]), [tf.shape(channelInt)[0], channelInt.shape[1]]), axis=2), 
                                                        indices <= tf.expand_dims(tf.reshape(tf.repeat(tf.cast(tf.math.ceil(predRTInds[:,1]), dtype=channelInt.dtype), repeats=[channelInt.shape[1]]), [tf.shape(channelInt)[0], channelInt.shape[1]]), axis=2)), 
                                    channelInt, tf.zeros_like(channelInt))
                maxRow = tf.where(stripped == 0, tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(channelInt.shape[1])), [tf.shape(channelInt)[0], channelInt.shape[1], channelInt.shape[2]]), stripped)
                minVal = tf.reduce_min(maxRow, axis=1)
                stripped = stripped - tf.reshape(tf.repeat(minVal, repeats=channelInt.shape[1]), [tf.shape(channelInt)[0], channelInt.shape[1], channelInt.shape[2]])
                stripped = tf.where(stripped < 0, tf.zeros_like(stripped), stripped)
                pbCalcArea = tf.squeeze(tf.reduce_sum(stripped, axis=1), axis=1)

                ## Extract area for overlap of user and PeakBot_MRM integration
                beginInds = tf.math.maximum(inteRTInds[:,0], tf.cast(tf.math.floor(predRTInds[:,0]), dtype=inteRTInds.dtype))
                endInds = tf.math.minimum(inteRTInds[:,1], tf.cast(tf.math.ceil(predRTInds[:,1]), dtype=predRTInds.dtype))
                stripped = tf.where(tf.math.logical_and(indices >= tf.expand_dims(tf.reshape(tf.repeat(beginInds, repeats=[channelInt.shape[1]]), [tf.shape(channelInt)[0], channelInt.shape[1]]), axis=2), 
                                                        indices <= tf.expand_dims(tf.reshape(tf.repeat(endInds, repeats=[channelInt.shape[1]]), [tf.shape(channelInt)[0], channelInt.shape[1]]), axis=2)), 
                                    channelInt, tf.zeros_like(channelInt))
                maxRow = tf.where(stripped == 0, tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(channelInt.shape[1])), tf.shape(channelInt)), stripped)
                minVal = tf.reduce_min(maxRow, axis=1)
                stripped = stripped - tf.reshape(tf.repeat(minVal, repeats=channelInt.shape[1]), [tf.shape(channelInt)[0], channelInt.shape[1], channelInt.shape[2]])
                stripped = tf.where(stripped < 0, tf.zeros_like(stripped), stripped)
                overlapArea = tf.squeeze(tf.reduce_sum(stripped, axis=1), axis=1)            
                #overlapArea = tf.where(tf.argmax(peaks, axis=1) == 1, tf.zeros_like(overlapArea), overlapArea)

                ## Calculate IOU
                overlap = tf.divide(overlapArea, tf.subtract(tf.add(inteArea, pbCalcArea), overlapArea))
                overlap = tf.keras.layers.Lambda(lambda x: x, name="")(overlap)
                loss = tf.square(overlap)
                return loss
            
            if mode == "training":
                self.model.add_loss(areaIOULoss(rtInds_, rtInds, input_))
                self.model.add_loss(tf.losses.mse(peaks_, peaks))
                self.model.add_metric(areaIOULoss(rtInds_, rtInds, input_), name="IOUArea")
        
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = Config.LEARNINGRATESTART),
                           loss={"pred.peak": "CategoricalCrossentropy",
                                 "pred.rtInds": "MSE"
                                 },
                           loss_weights={"pred.peak": 1,
                                         "pred.rtInds": 1/5000,},
                           metrics={"pred.peak": "categorical_accuracy",
                                    "pred.rtInds": iou,},
                                    #"loss.IOU_Area": "MSE"},
                           )
    
    @timeit
    def compileModel(self, learningRate = None):
        pass

    @timeit
    def train(self, datTrain, datVal, logDir = None, callbacks = None, verbose = 1):
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
            for metric in ["pred.peak_categorical_accuracy", "pred.rtInds_iou"]:
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

    assert all(np.amax(instances["channel.int"], (1)) == 1), "channel.int is not scaled to a maximum of 1 '%s'"%(str(np.amax(instances["channel.int"], (1))))
    
    peakTypes, rtInds = pb.model.predict(instances["channel.int"])
    rtStartInds = rtInds[:,0]
    rtEndInds = rtInds[:,1]

    return peakTypes, rtStartInds, rtEndInds




@timeit
def evaluatePeakBot(instancesWithGT, modelPath = None, model = None, verbose = True):
    tic("detecting with peakbot")

    if verbose:
        print("Evaluating peaks with PeakBot")
        print("  | .. loading PeakBot model '%s'"%(modelPath))

    pb = model
    if model is None and modelPath is not None:
        pb = loadModel(modelPath, mode = "training")

    assert all(np.amax(instancesWithGT["channel.int"], (1)) == 1), "channel.int is not scaled to a maximum of 1 '%s'"%(str(np.amax(instancesWithGT["channel.int"], (1))))
    
    x = {"channel.int": instancesWithGT["channel.int"],
         #"inte.peak"  : instancesWithGT["inte.peak"],
         #"inte.rtInds": instancesWithGT["inte.rtInds"],
        }
    y = {"pred.peak"  : instancesWithGT["inte.peak"],
         "pred.rtInds": instancesWithGT["inte.rtInds"],
         #"loss.IOU_Area": np.ones((len(instancesWithGT["inte.peak"])), dtype=float)
        }
    ret = pb.model.evaluate(x, y, return_dict = True, verbose = 0)

    return ret

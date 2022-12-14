from .core import *

import tensorflow as tf
import tensorflow_addons as tfa







## Custom metric functions 
## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05 - thanks a lot!
import numpy as np
import tensorflow as tf


class Accuracy4Peaks(tf.keras.metrics.Metric):

    def __init__(self, name='Acc4Peaks', **kwargs):
        super(Accuracy4Peaks, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.int32)
        self.count = self.add_weight(name='counts', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):    
        y_pred = tf.math.argmax(y_pred, axis=1)
        y_true = tf.math.argmax(y_true, axis=1)
        self.true_positives.assign_add(tf.cast(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.logical_and(y_pred == y_true, y_pred == 0))), dtype=tf.int32))
        self.count.assign_add(tf.cast(tf.math.reduce_sum(tf.math.count_nonzero(y_true == 0)), dtype=tf.int32))

    def result(self):
        temp = self.true_positives / self.count
        temp = tf.where(tf.math.is_nan(temp), tf.zeros_like(temp), temp)
        return temp

class Accuracy4NonPeaks(tf.keras.metrics.Metric):

    def __init__(self, name='Acc4NonPeaks', **kwargs):
        super(Accuracy4NonPeaks, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.int32)
        self.count = self.add_weight(name='counts', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):    
        y_pred = tf.math.argmax(y_pred, axis=1)
        y_true = tf.math.argmax(y_true, axis=1)
        self.true_positives.assign_add(tf.cast(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.logical_and(y_pred == y_true, y_pred == 1))), dtype=tf.int32))
        self.count.assign_add(tf.cast(tf.math.reduce_sum(tf.math.count_nonzero(y_true == 1)), dtype=tf.int32))

    def result(self):
        temp = self.true_positives / self.count
        temp = tf.where(tf.math.is_nan(temp), tf.zeros_like(temp), temp)
        return temp
        
    









#####################################
### PeakBot additional methods
##
@tf.autograph.experimental.do_not_convert
def _EICIOU(dummyX, dummyY, numClasses = None):
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]
    
    ## Calculate indices for EICs
    indices = tf.transpose(tf.reshape(tf.repeat(tf.range(tf.shape(eic)[1], dtype=eic.dtype), repeats=tf.shape(eic)[0]), [tf.shape(eic)[1], tf.shape(eic)[0]]))

    if PeakBotMRM.Config.UPDATEPEAKBORDERSTOMIN:
        ## Update borders to be min values in left and right side of the peak's apex
        stripped = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(prtInds[:,0]), repeats=tf.shape(eic)[1]), shape=tf.shape(eic))), 
                                                tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (prtInds[:,1]), repeats=tf.shape(eic)[1]), shape=tf.shape(eic)))), 
                            eic, 
                            tf.zeros_like(eic))
        maxInd   = tf.cast(tf.math.argmax(stripped, axis=1), dtype=tf.float32)
        
        strippeL = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(prtInds[:,0]), repeats=tf.shape(eic)[1]), shape=tf.shape(eic))), 
                                                tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (maxInd      ), repeats=tf.shape(eic)[1]), shape=tf.shape(eic)))), 
                            eic, 
                            tf.reshape(tf.repeat(tf.math.reduce_max(eic, axis=1), repeats=tf.shape(eic)[1]), shape=tf.shape(eic)))
        minIndL  = tf.reshape(tf.cast(tf.math.argmin(strippeL, axis=1), dtype=tf.float32), shape=(tf.shape(eic)[0],1))
        
        strippeR = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(maxInd      ), repeats=tf.shape(eic)[1]), shape=tf.shape(eic))), 
                                                tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (prtInds[:,1]), repeats=tf.shape(eic)[1]), shape=tf.shape(eic)))), 
                            eic, 
                            tf.reshape(tf.repeat(tf.math.reduce_max(eic, axis=1), repeats=tf.shape(eic)[1]), shape=tf.shape(eic)))
        minIndR  = tf.reshape(tf.cast(tf.math.argmin(strippeR, axis=1), dtype=tf.float32), shape=(tf.shape(eic)[0],1))
        prtInds  = tf.concat([minIndL, minIndR], axis=1)

    ## Extract area for user integration
    stripped = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(rtInds[:,0]), repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                            tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (rtInds[:,1]), repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                        eic, 
                        tf.zeros_like(eic))
    maxRow   = tf.where(tf.math.equal(stripped, 0), 
                        tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), tf.shape(eic)), 
                        stripped)
    minVal   = tf.reduce_min(maxRow, axis=1)
    stripped = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), tf.shape(eic)))
    stripped = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    inteArea = tf.reduce_sum(stripped, axis=1)

    ## Extract area for PeakBotMRM integration
    stripped   = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(prtInds[:,0]), repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                              tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (prtInds[:,1]), repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                          eic, 
                          tf.zeros_like(eic))
    maxRow     = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), tf.shape(eic)), stripped)
    minVal     = tf.reduce_min(maxRow, axis=1)
    stripped   = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), tf.shape(eic)))
    stripped   = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    pbCalcArea = tf.reduce_sum(stripped, axis=1)

    ## Extract area for overlap of user and PeakBotMRM integration
    beginInds   = tf.math.floor(tf.math.maximum(rtInds[:,0], prtInds[:,0]))
    endInds     = tf.math.ceil (tf.math.minimum(rtInds[:,1], prtInds[:,1]))
    stripped    = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(beginInds, repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                               tf.math.less_equal   (indices, tf.reshape(tf.repeat(endInds  , repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                           eic, 
                           tf.zeros_like(eic))
    maxRow      = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), tf.shape(eic)), stripped)
    minVal      = tf.reduce_min(maxRow, axis=1)
    stripped    = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), tf.shape(eic)))
    stripped    = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    overlapArea = tf.reduce_sum(stripped, axis=1)

    ## Calculate IOU
    iou = tf.divide(overlapArea + 0.0001, inteArea + pbCalcArea - overlapArea + 0.0001)

    return iou


class EICIOUPeaks(tf.keras.metrics.Metric):

    def __init__(self, name='EICIOUPeaks', numClasses = None, **kwargs):
        super(EICIOUPeaks, self).__init__(name=name, **kwargs)
        if numClasses is None:
            numClasses = PeakBotMRM.Config.NUMCLASSES
        self.numClasses = numClasses
        self.ious = self.add_weight(name='ious', initializer='zeros', dtype=tf.float32)
        self.counts = self.add_weight(name='counts', initializer='zeros', dtype=tf.float32)

    def update_state(self, dummyX, dummyY, sample_weight=None):        
        ## separate user integration and eic
        peaks   = dummyX[:, 0:self.numClasses]
        rtInds  = dummyX[:, self.numClasses:(self.numClasses + 2)]
        eic     = dummyX[:, (self.numClasses + 2):]
        
        ## separate predicted values
        ppeaks  = dummyY[:, 0:self.numClasses]
        prtInds = dummyY[:, self.numClasses:(self.numClasses + 2)]

        ## Calculate IOU
        iou = _EICIOU(dummyX, dummyY)
        ## set IOU to 0 if gt and prediction of peak do not match
        iou = tf.where(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), iou, tf.zeros_like(iou))
        ## Calculate IOU only for peaks
        iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iou, tf.zeros_like(iou))
        ## Calculate mean IOU and replace nan values with 0
        self.ious.assign_add(tf.reduce_sum(iou))
        self.counts.assign_add(tf.math.count_nonzero(iou, dtype=tf.float32))

    def result(self):
        temp = self.ious / self.counts
        temp = tf.where(tf.math.is_nan(temp), tf.zeros_like(temp), temp)
        return temp


@tf.autograph.experimental.do_not_convert
def EICIOULoss(dummyX, dummyY, numClasses = None):    
    
    '''    
    ## R-code to simulate the loss
    
    library(ggplot2)
    library(dplyr)
    library(magrittr)
    library(cowplot)

    start = 15
    end = 15.3

    rts = seq(15 - 1.4, end + 0.4, 0.01)

    start = rts[which.min(abs(rts - start))]
    end = rts[which.min(abs(rts - end))]

    eic = dnorm(rts, mean = (end + start) / 2, sd = 0.05)
    eic = eic + dnorm(rts, mean = 14.88, sd = 0.05) / 3
    plot(rts, eic)
    abline(v = c(start, end), col = "firebrick")


    dat = data.frame(type = c(), rt = c(), val = c())

    for(s in seq(start - 1.4, end , 0.01)){
    
    mse = (start - s) ^ 2
    dat = rbind(dat, data.frame(type = "MSE", rt = s, val = mse))
    
    t = (end - max(s,start)) / (end - min(s,start))
    dat = rbind(dat, data.frame(type = "IOU", rt = s, val = t))
    
    iou = sum(eic[which(rts == max(s,start)):which(rts == end)]) / sum(eic[which(rts == min(s,start)):which(rts == end)])
    dat = rbind(dat, data.frame(type = "AreaIOU", rt = s, val = iou))
    
    t = mse * (sqrt(1 - iou) + sqrt(mse) )
    dat = rbind(dat, data.frame(type = "AreaIOULoss", rt = s, val = t))
    }


    p1 <- ggplot(data = data.frame(rts = rts, eic = eic), mapping = aes(x = rts, y = eic)) +
    theme_minimal() +
    geom_line(size=2) +
    geom_vline(xintercept = c(start, end), colour = "firebrick", size = 2) +
    geom_vline(xintercept = c(start), colour = "firebrick", size = 3) +
    ggtitle("EIC")

    p2 <- ggplot(data = dat, mapping = aes(x = rt, y = val, colour = type)) +
    theme_minimal() +
    geom_line(size=2) +
    ggtitle("Overview of metrics") +
    theme(legend.position = "bottom")

    p3 <- ggplot(data = dat %>% dplyr::filter(type %in% c("MSE", "AreaIOULoss")), mapping = aes(x = rt, y = val, colour = type)) +
    theme_minimal() +
    geom_line(size=2) +
    ggtitle("Overview of metrics") +
    theme(legend.position = "bottom")

    p4 <- ggplot(data = dat %>% dplyr::filter(type %in% c("MSE", "AreaIOULoss"), rt > 14.8, rt < 15.25), mapping = aes(x = rt, y = val, colour = type)) +
    theme_minimal() +
    geom_line(size=2) +
    ggtitle("Overview of metrics") +
    theme(legend.position = "bottom")

    print(plot_grid(p1, p2, plot_grid(p3, p4, ncol=1), ncol=3))
    
    '''
    
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]

    ## get IOUloss 
    iouloss = 1 - _EICIOU(dummyX, dummyY)
    ## Only get IOUloss for true peaks integrated and predicted alike
    iouloss = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iouloss, tf.zeros_like(iouloss))
    ## Calculate MSE
    mse = tf.reduce_mean(tf.square(rtInds-prtInds), axis=1)
    ## Combine iou loss with MSE
    loss = mse * (tf.sqrt(tf.abs(iouloss)) + tf.sqrt(mse))
    loss = tf.where(iouloss > 0, loss, tf.zeros_like(iouloss))
    
    return loss

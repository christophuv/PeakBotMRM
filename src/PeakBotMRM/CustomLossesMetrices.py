from .core import *

import tensorflow as tf
import tensorflow_addons as tfa







## Custom metric functions 
## taken from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05 - thanks a lot!
import numpy as np
import tensorflow as tf
from keras import backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def fbeta(y_true, y_pred, beta=2):
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def equal_error_rate(y_true, y_pred):
    n_imp = tf.count_nonzero(tf.equal(y_true, 0), dtype=tf.float32) + tf.constant(K.epsilon())
    n_gen = tf.count_nonzero(tf.equal(y_true, 1), dtype=tf.float32) + tf.constant(K.epsilon())

    scores_imp = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
    scores_gen = tf.boolean_mask(y_pred, tf.equal(y_true, 1))

    loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
    cond = lambda t, fpr, fnr: tf.greater_equal(fpr, fnr)
    body = lambda t, fpr, fnr: (
        t + 0.001,
        tf.divide(tf.count_nonzero(tf.greater_equal(scores_imp, t), dtype=tf.float32), n_imp),
        tf.divide(tf.count_nonzero(tf.less(scores_gen, t), dtype=tf.float32), n_gen)
    )
    t, fpr, fnr = tf.while_loop(cond, body, loop_vars, back_prop=False)
    eer = (fpr + fnr) / 2

    return eer











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

    ## Extract area for user integration
    stripped = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(rtInds[:,0]), repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                            tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (rtInds[:,1]), repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                        eic, 
                        tf.zeros_like(eic))
    maxRow   = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), [tf.shape(eic)[0], tf.shape(eic)[1]]), stripped)
    minVal   = tf.reduce_min(maxRow, axis=1)
    stripped = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), [tf.shape(eic)[0], tf.shape(eic)[1]]))
    stripped = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    inteArea = tf.reduce_sum(stripped, axis=1)

    ## Extract area for PeakBotMRM integration
    stripped   = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(tf.math.floor(prtInds[:,0]), repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                            tf.math.less_equal   (indices, tf.reshape(tf.repeat(tf.math.ceil (prtInds[:,1]), repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                          eic, 
                          tf.zeros_like(eic))
    maxRow     = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), [tf.shape(eic)[0], tf.shape(eic)[1]]), stripped)
    minVal     = tf.reduce_min(maxRow, axis=1)
    stripped   = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), [tf.shape(eic)[0], tf.shape(eic)[1]]))
    stripped   = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    pbCalcArea = tf.reduce_sum(stripped, axis=1)

    ## Extract area for overlap of user and PeakBotMRM integration
    beginInds   = tf.math.floor(tf.math.maximum(rtInds[:,0], prtInds[:,0]))
    endInds     = tf.math.ceil (tf.math.minimum(rtInds[:,1], prtInds[:,1]))
    stripped    = tf.where(tf.math.logical_and(tf.math.greater_equal(indices, tf.reshape(tf.repeat(beginInds, repeats=tf.shape(eic)[1]), tf.shape(eic))), 
                                             tf.math.less_equal   (indices, tf.reshape(tf.repeat(endInds  , repeats=tf.shape(eic)[1]), tf.shape(eic)))), 
                           eic, 
                           tf.zeros_like(eic))
    maxRow      = tf.where(tf.math.equal(stripped, 0), tf.reshape(tf.repeat(tf.reduce_max(stripped, axis=1), repeats=(tf.shape(eic)[1])), [tf.shape(eic)[0], tf.shape(eic)[1]]), stripped)
    minVal      = tf.reduce_min(maxRow, axis=1)
    stripped    = tf.subtract(stripped, tf.reshape(tf.repeat(minVal, repeats=tf.shape(eic)[1]), [tf.shape(eic)[0], tf.shape(eic)[1]]))
    stripped    = tf.where(tf.math.less(stripped, 0), tf.zeros_like(stripped)+0.0001, stripped)
    overlapArea = tf.reduce_sum(stripped, axis=1)

    ## Calculate IOU
    iou = tf.divide(overlapArea + 0.0001, inteArea + pbCalcArea - overlapArea + 0.0001)

    return iou

@tf.autograph.experimental.do_not_convert
def EICIOU(dummyX, dummyY, numClasses = None):
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]

    ## Calculate IOU
    iou = _EICIOU(dummyX, dummyY)
    ## set IOU to 0 if gt and prediction of peak do not match
    iou = tf.where(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), iou, tf.zeros_like(iou))
    ## Calculate IOU only for peaks
    iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iou, tf.zeros_like(iou))
    
    return iou

@tf.autograph.experimental.do_not_convert
def EICIOUPeaks(dummyX, dummyY, numClasses = None):
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]

    ## Calculate IOU
    iou = _EICIOU(dummyX, dummyY)
    ## set IOU to 0 if gt and prediction of peak do not match
    iou = tf.where(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), iou, tf.zeros_like(iou))
    ## Calculate IOU only for peaks
    iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iou, tf.zeros_like(iou))
    ## Calculate mean IOU and replace nan values with 0
    iou = tf.reduce_sum(iou) / tf.cast(tf.math.count_nonzero(iou), dtype=iou.dtype)
    iou = tf.where(tf.math.is_nan(iou), tf.zeros_like(iou), iou)

    return iou

@tf.autograph.experimental.do_not_convert
def EICIOULoss(dummyX, dummyY, numClasses = None):
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
    iou = 1 - _EICIOU(dummyX, dummyY)
    ## Only get IOUloss for true peaks integrated and predicted alike
    iou = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), iou, tf.zeros_like(iou))
    ## Calculate MSE
    mse = tf.reduce_mean(tf.square(rtInds-prtInds), axis=1)
    ## Combine iou loss with MSE
    loss = mse * tf.sqrt(tf.abs(iou))
    
    return loss

@tf.autograph.experimental.do_not_convert
def CCAPeaks(dummyX, dummyY, numClasses = None):
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]
   
    ## Calculate CategoricalAccuracy
    cca = tf.keras.metrics.categorical_accuracy(peaks, ppeaks)
    
    return cca

@tf.autograph.experimental.do_not_convert
def MSERtInds(dummyX, dummyY, numClasses = None):
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]

    ## Calculate MSE
    mse = tf.keras.losses.MeanSquaredError()(rtInds, prtInds)
    
    return mse

@tf.autograph.experimental.do_not_convert
def MSERtIndsPeaks(dummyX, dummyY, numClasses = None):
    if numClasses is None:
        numClasses = PeakBotMRM.Config.NUMCLASSES
    
    ## separate user integration and eic
    peaks   = dummyX[:, 0:numClasses]
    rtInds  = dummyX[:, numClasses:(numClasses + 2)]
    eic     = dummyX[:, (numClasses + 2):]
    
    ## separate predicted values
    ppeaks  = dummyY[:, 0:numClasses]
    prtInds = dummyY[:, numClasses:(numClasses + 2)]
    
    ## Calculate MSE
    mse = tf.keras.losses.MeanSquaredError()(rtInds, prtInds)
    ## Remove MSE for non peaks
    mse = tf.where(tf.math.logical_and(tf.math.equal(tf.argmax(peaks, axis=1), tf.argmax(ppeaks, axis=1)), tf.argmax(peaks, axis=1) == 0), mse, tf.zeros_like(mse))
    
    return mse
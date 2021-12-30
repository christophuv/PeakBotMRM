from .core import *

import tensorflow as tf
import tensorflow_addons as tfa

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
                    try:
                        with file_writer.as_default():
                            tf.summary.scalar(valuename, data=result, step=epoch)
                    except Exception:
                        print("Cannot write scalar to tensorboard. Try installing TensorBoard (command: pip install tensorboard)")
                    if i > 0: outStr.append(", ")
                    valuename = metric
                    if i not in self.printWidths.keys():
                        self.printWidths[i] = 0
                    self.printWidths[i] = max(self.printWidths[i], len(valuename))
                    outStr.append("%s: %.4f"%("%%%ds"%self.printWidths[i]%valuename, result))
                    hist[validation_set_name + "_" + valuename] = result
                outStr.append("")
                outStr.insert(0, "   %%%ds - %d instances - %3.0fs - "%(self.maxLenNames, validation_data["channel.int"].shape[0], toc("kl234hlkjsfkjh1hlkjhasfdkjlh"))%validation_set_name)
                if self.verbose: print("".join(outStr))
            if self.verbose: print("")
        self.history.append(hist)

    def on_train_begin(self, logs=None):
        self.on_epoch_end(self.lastEpochNum, logs = logs, ignoreEpoch = True)

    def on_train_end(self, logs=None):
        self.on_epoch_end(self.lastEpochNum, logs = logs, ignoreEpoch = True)
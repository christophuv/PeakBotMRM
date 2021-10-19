
from peakbot_MRM.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary
import peakbot_MRM

import os
import pathlib
import pickle
import tqdm
import random
import math
import shutil
import pathlib
    

import numpy as np


def shuffleResultsSampleNames(exportPath, instancePrefix=None,
                              tempFileName="bqu40mcb25473zfhbgwh22534", verbose=False):

    if instancePrefix is None:
        instancePrefix = peakbot_MRM.Config.INSTANCEPREFIX

    tic("shuffling")
    if verbose:
        print("Shuffling the test instances (batch name shuffling)")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose:
        print("  | .. there are %d files" % (len(files)))

    unused = [t for t in range(len(files))]
    random.shuffle(unused)
    for i in range(len(files)):
        j = unused[0]
        del unused[0]

        os.rename(files[i], os.path.join(pathlib.Path(files[i]).parent.resolve(), "%s%d.pickle" % (tempFileName, i)))

    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    for i in range(len(files)):
        os.rename(files[i], files[i].replace(tempFileName, instancePrefix))

    if verbose:
        print("  | .. took %.1f seconds" % toc("shuffling"))
        print("")


def shuffleResults(exportPath, steps=1E5, samplesToExchange=50,
                   instancePrefix=None, verbose=False):

    if instancePrefix is None:
        instancePrefix = peakbot_MRM.Config.INSTANCEPREFIX

    tic("shuffling")
    if verbose:
        print("Shuffling the test instances (inter-batch shuffling)")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose:
        print("  | .. there are %d files" % (len(files)))

    with tqdm.tqdm(total=steps, desc="  | .. shuffling", disable=not verbose) as t:
        while steps > 0:
            filea = files[random.randint(0, len(files) - 1)]
            fileb = files[random.randint(0, len(files) - 1)]

            if filea == fileb:
                continue

            with open(filea, "rb") as temp:
                a = pickle.load(temp)
            with open(fileb, "rb") as temp:
                b = pickle.load(temp)

            samplesA = a["channel.rt"].shape[0]
            samplesB = b["channel.rt"].shape[0]

            cExchange = min(min(samplesA, samplesB), samplesToExchange)

            beginA = random.randint(0, samplesA - cExchange)
            beginB = random.randint(0, samplesB - cExchange)

            for k in a.keys():
                if isinstance(a[k], np.ndarray) and len(a[k].shape) == 1:
                    temp = a[k][beginA:(beginA + cExchange)]
                    a[k][beginA:(beginA + cExchange)] = b[k][beginB:(beginB + cExchange)]
                    b[k][beginB:(beginB + cExchange)] = temp

                elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 2:
                    temp = a[k][beginA:(beginA + cExchange), :]
                    a[k][beginA:(beginA + cExchange),:] = b[k][beginB:(beginB + cExchange), :]
                    b[k][beginB:(beginB + cExchange), :] = temp

                elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 3:
                    temp = a[k][beginA:(beginA + cExchange), :, :]
                    a[k][beginA:(beginA + cExchange), :,:] = b[k][beginB:(beginB + cExchange), :, :]
                    b[k][beginB:(beginB + cExchange), :, :] = temp

                elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 4:
                    temp = a[k][beginA:(beginA + cExchange), :, :, :]
                    a[k][beginA:(beginA + cExchange), :, :,:] = b[k][beginB:(beginB + cExchange), :, :, :]
                    b[k][beginB:(beginB + cExchange), :, :, :] = temp

                elif isinstance(a[k], list):
                    temp = a[k][beginA:(beginA + cExchange)]
                    a[k][beginA:(beginA + cExchange)] = b[k][beginB:(beginB + cExchange)]
                    b[k][beginB:(beginB + cExchange)] = temp
                
                else:
                    assert False, "Unknown key in shuffling, aborting"

            assert samplesA == a["channel.rt"].shape[0] and samplesB == b["channel.rt"].shape[0]

            with open(filea, "wb") as temp:
                pickle.dump(a, temp)
            with open(fileb, "wb") as temp:
                pickle.dump(b, temp)

            steps = steps - 1
            t.update()

    if verbose:
        print("  | .. took %.1f seconds" % toc("shuffling"))
        print("")


def splitDSinto(path, newDS1Path, newDS2Path = None, ratioDS1 = 0.3, instancePrefix = None, tempFileName = "bqu40mcb25473zfhbgwh22534", copy=False, verbose = False):

    assert 0 <= ratioDS1 <= 1, "parameter ratioDS1 must be 0 <= ratioDS1 <= 1"
    
    pathlib.Path(newDS1Path).mkdir(parents=True, exist_ok=True) 
    if newDS2Path is not None:
        pathlib.Path(newDS2Path).mkdir(parents=True, exist_ok=True) 

    if instancePrefix is None:
        instancePrefix = peakbot_MRM.Config.INSTANCEPREFIX

    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    take = math.floor(len(files)*ratioDS1)

    cur = 0
    while take > 0:

        randFile = random.randint(0, len(files)-1)
        if copy:
            shutil.copy(files[randFile], os.path.join(newDS1Path, "%s%d.pickle"%(instancePrefix, cur)))
        else:
            shutil.move(files[randFile], os.path.join(newDS1Path, "%s%d.pickle"%(instancePrefix, cur)))
        
        del files[randFile]
        take = take - 1
        cur = cur + 1

    if newDS2Path is not None:
        temp = 0
        for fil in files:
            if copy:
                shutil.copy(fil, os.path.join(newDS2Path, "%s%d.pickle"%(instancePrefix, temp)))
            else:
                shutil.move(fil, os.path.join(newDS2Path, "%s%d.pickle"%(instancePrefix, temp)))
            temp = temp + 1

    if not copy:    
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for i in range(len(files)):
            os.rename(files[i], os.path.join(pathlib.Path(files[i]).parent.resolve(), "%s%d.pickle" % (tempFileName, i)))
        
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for i in range(len(files)):
            os.rename(files[i], os.path.join(pathlib.Path(files[i]).parent.resolve(), "%s%d.pickle" % (instancePrefix, i)))

    if verbose:
        print("  | .. %s %d files from the dataset '' (now %d instances) to the new dataset ''"%("copied" if copy else "moved", cur, path, newDS1Path))
        if newDS2Path is not None:
            print("  | .. %s remaining instances to '%s'"%("copied" if copy else "moved", newDS2Path))

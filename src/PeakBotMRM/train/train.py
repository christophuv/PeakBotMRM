from PeakBotMRM.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary
import PeakBotMRM
from PeakBotMRM.core import readTSVFile, parseTSVMultiLineHeader, extractStandardizedEIC, scaleStandardizedEIC, getInteRTIndsOnStandardizedEIC

import os
import pathlib
import tempfile
import matplotlib.pyplot as plt
import plotnine as p9
import pandas as pd
import numpy as np
import pickle
import tqdm
import random
random.seed(2021)
import math
import shutil
import pathlib




def shuffleResultsSampleNames(exportPath, instancePrefix=None,
                              tempFileName="bqu40mcb25473zfhbgwh22534", verbose=True):

    if instancePrefix is None:
        instancePrefix = PeakBotMRM.Config.INSTANCEPREFIX

    tic("shuffling")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose:
        print("  | .. shuffling the test instances (batch name shuffling), there are %d files" % (len(files)))

    random.shuffle(files)
    for i in range(len(files)):
        os.rename(files[i], os.path.join(pathlib.Path(files[i]).parent.resolve(), "%s%d.pickle" % (tempFileName, i)))


    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    for i in tqdm.tqdm(range(len(files)), desc="  | .. shuffling file names", disable = not verbose):
        os.rename(files[i], files[i].replace(tempFileName, instancePrefix))


def shuffleResults(exportPath, steps=1E5, samplesToExchange=50,
                   instancePrefix=None, verbose=True):

    if instancePrefix is None:
        instancePrefix = PeakBotMRM.Config.INSTANCEPREFIX

    tic("shuffling")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose:
        print("  | .. shuffling the test instances (inter-batch shuffling), there are %d files" % (len(files)))
            

    with tqdm.tqdm(total=steps, desc="  | .. shuffling instances", disable=not verbose) as t:
        while steps > 0:
            a = None
            b = None

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

            cExchange = math.floor(min(min(samplesA/2, samplesB/2), samplesToExchange))

            beginA = random.randint(0, samplesA - cExchange)
            beginB = random.randint(0, samplesB - cExchange)

            if cExchange > 0:
                assert beginA >= 0 and beginB >= 0 and (beginA + cExchange) <= samplesA and (beginB + cExchange) <= samplesB and filea != fileb

                for k in a.keys():
                    if   isinstance(a[k], np.ndarray) and len(a[k].shape) == 1:
                        a[k][beginA:(beginA + cExchange)],       b[k][beginB:(beginB + cExchange)]       = np.copy(b[k][beginB:(beginB + cExchange)]),       np.copy(a[k][beginA:(beginA + cExchange)])

                    elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 2:
                        a[k][beginA:(beginA + cExchange),:],     b[k][beginB:(beginB + cExchange),:]     = np.copy(b[k][beginB:(beginB + cExchange),:]),     np.copy(a[k][beginA:(beginA + cExchange),:])

                    elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 3:
                        a[k][beginA:(beginA + cExchange),:,:],   b[k][beginB:(beginB + cExchange),:,:]   = np.copy(b[k][beginB:(beginB + cExchange),:,:]),   np.copy(a[k][beginA:(beginA + cExchange),:,:])

                    elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 4:
                        a[k][beginA:(beginA + cExchange),:,:,:], b[k][beginB:(beginB + cExchange),:,:,:] = np.copy(b[k][beginB:(beginB + cExchange),:,:,:]), np.copy(a[k][beginA:(beginA + cExchange),:,:,:])

                    elif isinstance(a[k], list):
                        a[k][beginA:(beginA + cExchange)],       b[k][beginB:(beginB + cExchange)]       = b[k][beginB:(beginB + cExchange)],                a[k][beginA:(beginA + cExchange)]
                        
                    else:
                        assert False, "Unknown key in shuffling, aborting"

                assert samplesA == a["channel.rt"].shape[0] and samplesB == b["channel.rt"].shape[0]

                with open(filea, "wb") as temp:
                    pickle.dump(a, temp)
                with open(fileb, "wb") as temp:
                    pickle.dump(b, temp)

            steps = steps - 1
            t.update()


def splitDSinto(path, newDS1Path = None, newDS2Path = None, ratioDS1 = 0.3, instancePrefix = None, tempFileName = "bqu40mcb25473zfhbgwh22534", copy=False, verbose = False):

    assert 0 <= ratioDS1 <= 1, "parameter ratioDS1 must be 0 <= ratioDS1 <= 1"
    
    if newDS1Path is not None:
        pathlib.Path(newDS1Path).mkdir(parents=True, exist_ok=True) 
    if newDS2Path is not None:
        pathlib.Path(newDS2Path).mkdir(parents=True, exist_ok=True) 

    if instancePrefix is None:
        instancePrefix = PeakBotMRM.Config.INSTANCEPREFIX

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

def splitDSAccordingToClass(path, newDS1Path, newDS2Path, instancePrefix = None, copy = True, verbose = False):
    
    if newDS1Path is not None:
        pathlib.Path(newDS1Path).mkdir(parents=True, exist_ok=True) 
    if newDS2Path is not None:
        pathlib.Path(newDS2Path).mkdir(parents=True, exist_ok=True) 

    if instancePrefix is None:
        instancePrefix = PeakBotMRM.Config.INSTANCEPREFIX

    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for fil in files:
        pass
    ## TODO finish function


def trainPeakBotMRMModel(expName, targetFile, curatedPeaks, samplesPath, modelFile, expDir = None, logDir = None, historyObject = None,
                         MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                         allowedMZOffset = 0.05, drawRawData = False,
                         addRandomNoise = True, maxRandFactor = 0.1, maxNoiseLevelAdd=0.1, shiftRTs = True, maxShift = 0.15, useEachInstanceNTimes = 5, 
                         excludeSubstances = None, includeSubstances = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if historyObject is None:
        historyObject = os.path.join(expDir, "History.pandas.pickle")
    if excludeSubstances is None:
        excludeSubstances = []

        
    print("Training model from experiment")
    print("  | .. Parameters")
    print("  | .. .. expName: '%s'"%(expName))
    print("  | .. .. targetFile: '%s'"%(targetFile))
    print("  | .. .. curatedPeaks: '%s'"%(curatedPeaks))
    print("  | .. .. samplesPath: '%s'"%(samplesPath))
    print("  | .. .. modelFile: '%s'"%(modelFile))
    print("  | .. .. expDir: '%s'"%(expDir))
    print("  | .. .. logDir: '%s'"%(logDir))
    print("  | .. .. MRMHeader: '%s'"%(MRMHeader))
    print("  | .. .. allowedMZOffset: '%s'"%(allowedMZOffset))
    print("  | .. .. addRandomNoise: '%s'"%(addRandomNoise))
    if addRandomNoise:
        print("  | .. .. maxRandFactor: '%s'"%(maxRandFactor))
    print("  | .. .. shiftRTs: '%s'"%(shiftRTs))
    if shiftRTs:
        print("  | .. .. maxShift: '%s'"%(maxShift))        
    print("\n")

    
    print("PeakBotMRM configuration")
    print(PeakBotMRM.Config.getAsStringFancy())
    print("\n")
    

    ## administrative
    tic("Overall process")
    try: 
        os.remove(historyObject)
    except:
        pass
    try:
        os.mkdir(expDir)
    except:
        pass
    try:
        os.mkdir(os.path.join(expDir, "SubstanceFigures"))
    except:
        pass        
    histAll = None


    substances               = PeakBotMRM.loadTargets(targetFile, excludeSubstances = excludeSubstances, includeSubstances = includeSubstances)
    substances, integrations = PeakBotMRM.loadIntegrations(substances, curatedPeaks)
    substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, samplesPath, expDir,
                                                             allowedMZOffset = allowedMZOffset, 
                                                             MRMHeader = MRMHeader)

    print("Balancing training dataset")
    tic()
    peaks = []
    noPeaks = []
    for substance in substances.values():
        if substance["Name"] in integrations.keys():
            for sample in integrations[substance["Name"]].keys():
                if integrations[substance["Name"]][sample]["foundPeak"]:
                    peaks.append((substance["Name"], sample))
                else:
                    noPeaks.append((substance["Name"], sample))
    print("  | .. there are %d peaks and %d backgrounds in the dataset"%(len(peaks), len(noPeaks)))
    random.shuffle(peaks)
    random.shuffle(noPeaks)
    a = min(len(peaks), len(noPeaks))
    peaks = peaks[:a]
    noPeaks = noPeaks[:a]
    inte2 = {}
    for substance, sample in peaks:
        if substance not in inte2.keys():
            inte2[substance] = {}
        inte2[substance][sample] = integrations[substance][sample]
        
    for substance, sample in noPeaks:
        if substance not in inte2.keys():
            inte2[substance] = {}
        inte2[substance][sample] = integrations[substance][sample]
    integrations = inte2

    peaks = []
    noPeaks = []
    for substance in substances.values():
        if substance["Name"] in integrations.keys():
            for sample in integrations[substance["Name"]].keys():
                if integrations[substance["Name"]][sample]["foundPeak"]:
                    peaks.append((substance["Name"], sample))
                else:
                    noPeaks.append((substance["Name"], sample))
    print("  | .. balanced dataset to %d peaks and %d backgrounds"%(len(peaks), len(noPeaks)))
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")


    if drawRawData:
        tic("drawEICs")
        offset = 0.2
        print("Exporting illustrations of data")
        for substance in tqdm.tqdm(integrations.keys(), desc="  | .. exporting"):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=False)
            fig.set_size_inches(15, 8)

            nPeaks = 0
            nNoPeaks = 0
            for samplei, sample in enumerate(integrations[substance].keys()):
                assert len(integrations[substance][sample]["chrom"]) <= 1
                
                if len(integrations[substance][sample]["chrom"]) == 1:
                    foundPeak, rtStart, rtEnd, area = integrations[substance][sample]["foundPeak"], integrations[substance][sample]["rtstart"], integrations[substance][sample]["rtend"], integrations[substance][sample]["area"]
                    inte = integrations[substance][sample]["chrom"][0]
                    chrom = inte[9]
                    
                    if foundPeak:
                        ax1.plot([t[0] for t in chrom if t[0]<=rtStart], [t[1] for t in chrom if t[0]<=rtStart], label=sample, linewidth=.5, alpha=.3, color="slategrey")
                        ax1.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [t[1] for t in chrom if rtStart<=t[0]<=rtEnd], label=sample, linewidth=2, alpha=.3, color="firebrick")
                        ax1.plot([t[0] for t in chrom if rtEnd<=t[0]], [t[1] for t in chrom if rtEnd<=t[0]], label=sample, linewidth=.5, alpha=.3, color="slategrey")

                        temp = [t[1] for t in chrom if rtStart<=t[0]<=rtEnd]
                        if len(temp) > 0:
                            ax2.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [t[1] for t in chrom if rtStart<=t[0]<=rtEnd], label=sample, linewidth=.5, alpha=.3, color="firebrick")
                            b = min(temp)
                            m = max([i-b for i in temp])
                            #ax3.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [(t[1]-b)/m for t in chrom if rtStart<=t[0]<=rtEnd], label=sample, linewidth=.5, alpha=.3, color="firebrick")
                        
                            ax3.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [(t[1]-b)/m+offset*samplei for t in chrom if rtStart<=t[0]<=rtEnd], "k", linewidth=.5, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                            ax3.fill_between([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [(t[1]-b)/m+offset*samplei for t in chrom if rtStart<=t[0]<=rtEnd], offset*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                        nPeaks +=1
                    else:
                        nNoPeaks += 1

                        ax4.plot([t[0] for t in chrom], [t[1] for t in chrom], label=sample, linewidth=.5, alpha=.3, color="slategrey")

                        temp = [t[1] for t in chrom]
                        b = min(temp)
                        m = max([i-b for i in temp])
                        ax5.plot([t[0] for t in chrom], [(t[1]-b)/m+offset*samplei for t in chrom], 'k', linewidth=.5, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ax5.fill_between([t[0] for t in chrom], [(t[1]-b)/m+offset*samplei for t in chrom], offset*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)

            ax1.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax2.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax3.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax4.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax5.axvline(x=substances[substance]["RT"], zorder=1E6)

            ax1.set(xlabel='time (min)', ylabel='abundance', title='Peaks\nRaw')
            ax2.set(xlabel='time (min)', ylabel='abundance', title='%s (%d peaks, %d no peaks)\nZoomed'%(substance, nPeaks, nNoPeaks))
            ax3.set(xlabel='time (min)', ylabel='abundance', title='\nNormalized')
            ax4.set(xlabel='time (min)', ylabel='abundance', title='No peaks\nRaw')
            ax5.set(xlabel='time (min)', ylabel='abundance', title='\nNormalized')

            ax3.set_ylim(-0.2, len(integrations[substance].keys())*offset+1+0.2)
            ax5.set_ylim(-0.2, len(integrations[substance].keys())*offset+1+0.2)
            
            plt.tight_layout()
            fig.savefig(os.path.join(expDir, "%s.png"%substance), dpi=300)
            plt.close()
        print("  | .. took %.1f seconds"%toc("drawEICs"))
        print("\n")


    print("Exporting instances for training")
    tic()
    instanceDirObj = tempfile.TemporaryDirectory()
    instanceDir = instanceDirObj.name
    if os.path.isfile(os.path.join(expDir, "trainingInstances.zip")):
        shutil.unpack_archive(os.path.join(expDir, "trainingInstances.zip"), instanceDir)
        print("  | .. Imported training instances from '%s'", os.path.join(expDir, "trainingInstances.zip"))
    else:
        temp = None
        cur = 0
        curI = 0
        ## iterate all samples and substances
        if addRandomNoise:
            print("  | .. Random noise will be added. The range of the randomly generated factors is %.3f - %.3f and the maximum randomly-generated noise added on top of the EICs is %.3f"%(1/(1 + maxRandFactor), 1 + maxRandFactor, maxNoiseLevelAdd))
        if shiftRTs:
            print("  | .. Random RT shifts will be added. The range is -%.3f - %.3f minutes"%(maxShift, maxShift))
            print("  | .. Chromatographic peaks with a shifted peak apex will first be corrected to the designated RT and then randomly moved for the training instance")
        if useEachInstanceNTimes > 1:
            print("  | .. Each instance will be used %d times augmented with different randomness factors"%(useEachInstanceNTimes))
        for substance in tqdm.tqdm(integrations.keys(), desc="  | .. augmenting"):
            for sample in integrations[substance].keys():
                for repi in range(useEachInstanceNTimes):
                    inte = integrations[substance][sample]
                    if len(inte["chrom"]) == 1:
                        
                        rts = inte["chrom"][0][9]["rts"]
                        eic = inte["chrom"][0][9]["eic"]
                        refRT = substances[substance]["RT"]

                        ## add uniform Rt shift to EICs
                        artificialRTShift = 0
                        if shiftRTs:
                            if inte["foundPeak"]:
                                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                                eicS[rtsS < inte["rtstart"]] = 0
                                eicS[rtsS > inte["rtend"]] = 0
                                apexRT = rtsS[np.argmax(eicS)]
                                artificialRTShift = apexRT - refRT + np.random.rand(1) * 2 * maxShift - maxShift
                            else:
                                artificialRTShift = np.random.rand(1) * 2 * maxShift - maxShift
                        
                        ## standardize EIC
                        rtsS, eicS = extractStandardizedEIC(eic, rts, refRT + artificialRTShift)
                        
                        ## get integration results on standardized area
                        bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                            getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                            inte["foundPeak"], 
                                                            inte["rtstart"], 
                                                            inte["rtend"])
                        
                        ## add random noise
                        if addRandomNoise:
                            ## randomize signal intensitiers
                            if np.random.rand(1)[0] > 0.5:
                                eicS = eicS * (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                            else:
                                eicS = eicS / (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                            
                            ## add noise on top of EIC
                            eicS = eicS + np.random.rand(eicS.shape[0]) * maxNoiseLevelAdd
                            
                            eicS = scaleStandardizedEIC(eicS, rtsS)
                        
                        ## test if eic has detected signals
                        if np.sum(eicS) > 0 and np.all(eicS >= 0):
                            ## add instance to training data
                            if temp is None:
                                temp = {"channel.rt"        : np.zeros((PeakBotMRM.Config.BATCHSIZE, PeakBotMRM.Config.RTSLICES), dtype=float),
                                        "channel.int"       : np.zeros((PeakBotMRM.Config.BATCHSIZE, PeakBotMRM.Config.RTSLICES), dtype=float),
                                        "inte.peak"         : np.zeros((PeakBotMRM.Config.BATCHSIZE, PeakBotMRM.Config.NUMCLASSES), dtype=int),
                                        "inte.rtStart"      : np.zeros((PeakBotMRM.Config.BATCHSIZE), dtype=float),
                                        "inte.rtEnd"        : np.zeros((PeakBotMRM.Config.BATCHSIZE), dtype=float),
                                        "inte.rtInds"       : np.zeros((PeakBotMRM.Config.BATCHSIZE, 2), dtype=float),
                                        "inte.area"         : np.zeros((PeakBotMRM.Config.BATCHSIZE), dtype=float),
                                        "ref.substanceName" : ["" for i in range(PeakBotMRM.Config.BATCHSIZE)],
                                        "ref.sample"        : ["" for i in range(PeakBotMRM.Config.BATCHSIZE)],
                                        "ref.rt"            : np.zeros((PeakBotMRM.Config.BATCHSIZE), dtype=float),
                                        "ref.PeakForm"      : ["" for i in range(PeakBotMRM.Config.BATCHSIZE)], 
                                        "ref.Rt shifts"     : ["" for i in range(PeakBotMRM.Config.BATCHSIZE)],
                                        "ref.Note"          : ["" for i in range(PeakBotMRM.Config.BATCHSIZE)],
                                        "loss.IOU_Area"     : np.ones((PeakBotMRM.Config.BATCHSIZE), dtype=float),}
                                curI = 0

                            assert curI < temp["channel.rt"].shape[0]
                            peakType = 0 if inte["foundPeak"] else 1

                            ## substance data
                            temp["ref.substanceName"][curI] = substance
                            temp["ref.sample"       ][curI] = sample
                            temp["ref.rt"           ][curI] = substances[substance]["RT"]
                            temp["ref.PeakForm"     ][curI] = substances[substance]["PeakForm"] 
                            temp["ref.Rt shifts"    ][curI] = substances[substance]["Rt shifts"]
                            temp["ref.Note"         ][curI] = substances[substance]["Note"]

                            ## analytical raw data
                            temp["channel.rt"       ][curI,:] = rtsS
                            temp["channel.int"      ][curI,:] = eicS

                            ## manual integration data
                            temp["inte.peak"        ][curI, peakType] = 1
                            temp["inte.rtStart"     ][curI]   = bestRTStart
                            temp["inte.rtEnd"       ][curI]   = bestRTEnd
                            temp["inte.rtInds"      ][curI,0] = bestRTStartInd
                            temp["inte.rtInds"      ][curI,1] = bestRTEndInd                        
                            temp["inte.area"        ][curI]   = inte["area"]

                            curI = curI + 1
                        else:
                            np.set_printoptions(edgeitems=PeakBotMRM.Config.RTSLICES + 2, 
                                formatter=dict(float=lambda x: "%.3g" % x))
                            print(eicS)

                    ## if batch has been filled, export it to a temporary file
                    if curI >= PeakBotMRM.Config.BATCHSIZE:
                        with open(os.path.join(instanceDir, "%s%d.pickle"%(PeakBotMRM.Config.INSTANCEPREFIX, cur)), "wb") as fout:
                            pickle.dump(temp, fout)
                            temp = None
                            curI = 0
                            cur += 1
        shutil.make_archive(os.path.join(expDir, "trainingInstances"), "zip", instanceDir)
    print("  | .. Exported batches each with %d instances (saved to '%s')."%(PeakBotMRM.Config.BATCHSIZE, os.path.join(expDir, "trainingInstances.zip")))

    ## randomize instances
    PeakBotMRM.train.shuffleResultsSampleNames(instanceDir)
    PeakBotMRM.train.shuffleResults(instanceDir, steps=1E4, samplesToExchange=12)
    print("  | .. Instances shuffled")
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")


    ## train new model
    with tempfile.TemporaryDirectory(prefix="PBMRM_training__") as tempTrainDir, tempfile.TemporaryDirectory(prefix="PBMRM_validation__") as tempValDir, tempfile.TemporaryDirectory(prefix="PBMRM_validationPeak__") as tempValDirClassPeak, tempfile.TemporaryDirectory(prefix="PBMRM_validationBackground__") as tempValDirClassBackground:
        PeakBotMRM.train.splitDSinto(instanceDir, 
                                     newDS1Path = tempTrainDir, newDS2Path = tempValDir, 
                                     copy = True, ratioDS1 = 0.7, verbose = False)
        PeakBotMRM.train.splitDSAccordingToClass(instanceDir, 
                                                 newDS1Path = tempTrainDir, newDS2Path = tempValDir, 
                                                 copy = True, verbose = False)
        
        nTrainBatches = len([f for f in os.listdir(tempTrainDir) if os.path.isfile(os.path.join(tempTrainDir, f))])
        nValBatches = len([f for f in os.listdir(tempValDir) if os.path.isfile(os.path.join(tempValDir, f))])         
        print("Split dataset")  
        tic()
        print("  | .. Randomly split dataset '%s' into a training and validation dataset with 0.7 and 0.3 parts of the instances "%expDir)
        print("  | .. There are %d training (%s) and %d validation (%s) batches available"%(nTrainBatches, tempTrainDir, nValBatches, tempValDir))
        print("  | .. With the current configuration (%d batchsize, %d steps per epoch) this will allow for %d epochs of training"%(PeakBotMRM.Config.BATCHSIZE, PeakBotMRM.Config.STEPSPEREPOCH, math.floor(nTrainBatches/PeakBotMRM.Config.STEPSPEREPOCH)))
        print("  | .. took %.1f seconds"%(toc()))
        print("\n")
        
        addValDS = []
        addValDS.append({"folder": tempTrainDir, "name": "train", "numBatches": nTrainBatches-1})            
        addValDS.append({"folder": tempValDir  , "name": "val"  , "numBatches": nValBatches  -1})

        pb, hist = PeakBotMRM.trainPeakBotMRMModel(trainInstancesPath = tempTrainDir,
                                                   addValidationInstances = addValDS,
                                                   logBaseDir = logDir,
                                                   everyNthEpoch = -1, 
                                                   verbose = True)

        pb.saveModelToFile(modelFile)
        print("Newly trained PeakBotMRM saved to file '%s'"%(modelFile))

        if histAll is None:
            histAll = hist
        else:
            histAll = histAll.append(hist, ignore_index=True)

        print("")
        print("")

        ### Summarize the training and validation metrices and losses
        histAll.to_pickle(historyObject)        

        ### Summarize and illustrate the results of the different training and validation dataset
        df = histAll
        df['ID'] = df.model.str.split('_').str[-1]
        df = df[df["metric"]!="loss"]
        df = df[df["set"]!="eV"]
        plot = (p9.ggplot(df, p9.aes("set", "value", colour="set"))
                + p9.geom_violin()
                + p9.geom_jitter()
                + p9.facet_wrap("~metric", scales="free_y", ncol=2)
                + p9.scale_x_discrete(limits=["train", "val"])
                + p9.ggtitle("Training losses/metrics") + p9.xlab("Training/Validation dataset") + p9.ylab("Value")
                + p9.theme(legend_position = "none", panel_spacing_x=0.5))
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "fig_SummaryPlot.png"), width=5.2, height=5, dpi=300)
    print("\n\n\n\n\n")

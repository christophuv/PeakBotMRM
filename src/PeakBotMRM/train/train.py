from PeakBotMRM.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary
import PeakBotMRM
from PeakBotMRM.core import readTSVFile, parseTSVMultiLineHeader, extractStandardizedEIC, getInteRTIndsOnStandardizedEIC

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
    random.shuffle(files)
    for i in range(len(files)):
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
            
            ministeps = math.floor(min(samplesA, samplesB) / min(min(samplesA/2, samplesB/2), samplesToExchange)) + 1
            
            while ministeps > 0 and steps > 0:

                cExchange = random.randint(1, min(min(samplesA/2, samplesB/2), samplesToExchange))

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

                    steps = steps - 1
                    ministeps = ministeps - 1
                    t.update()

            with open(filea, "wb") as temp:
                pickle.dump(a, temp)
            with open(fileb, "wb") as temp:
                pickle.dump(b, temp)


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
            
def showSampleOverview(instanceDir):
    files = [os.path.join(instanceDir, f) for f in os.listdir(instanceDir) if os.path.isfile(os.path.join(instanceDir, f))]
    
    peaks, backgrounds = 0, 0
    samplesUsed = {}
    substancesUsed = {}
    
    for f in files:
        with open(f, "rb") as temp:
            a = pickle.load(temp)
            
            peaks = peaks + np.sum(a["inte.peak"][:,0])
            backgrounds +=  np.sum(a["inte.peak"][:,0])
            
            for s in a["ref.sample"]:
                if s not in samplesUsed.keys():
                    samplesUsed[s] = 0
                samplesUsed[s] = samplesUsed[s] + 1
            for s in a["ref.substanceName"]:
                if s not in substancesUsed.keys():
                    substancesUsed[s] = 0
                substancesUsed[s] = substancesUsed[s] + 1
                
    print("  | .. .. There are %d peaks and %d peackgrounds in the dataset. "%(peaks, backgrounds))
    print("  | .. .. samples used:")
    for s in sorted(samplesUsed.keys()):
        print("  | .. .. .. %s: %d"%(s, samplesUsed[s]))
    print("  | .. .. substances used:")
    for s in sorted(substancesUsed.keys()):
        print("  | .. .. .. %s: %d"%(s, substancesUsed[s]))
                
            

def compileInstanceDataset(expDir, substances, integrations, instanceDir, addRandomNoise=False, maxRandFactor=0.1, maxNoiseLevelAdd=0.1, shiftRTs=False, maxShift=0.1, useEachInstanceNTimes=1, exportBatchSize = 1024, includeMetaInfo = False):
    temp = None
    cur = 0
    curI = 0
    ## iterate all samples and substances
    if addRandomNoise:
        print("  | .. Random noise will be added. The range of the randomly generated factors is %.3f - %.3f and the maximum randomly-generated noise added on top of the EICs is %.3f"%(1/(1 + maxRandFactor), 1 + maxRandFactor, maxNoiseLevelAdd))
    if shiftRTs:
        print("  | .. Random RT shifts will be added. The range is -%.3f - %.3f minutes"%(maxShift, maxShift))
        print("  | .. Chromatographic peaks with a shifted peak apex will first be corrected to the designated RT and then randomly moved for the training instance")
    print("  | .. Each instance will be used %d times"%(useEachInstanceNTimes))
    for substance in tqdm.tqdm(integrations.keys(), desc="  | .. augmenting"):
        for sample in integrations[substance].keys():
            inte = integrations[substance][sample]
            if len(inte["chrom"]) == 1:
                rts = inte["chrom"][0][9]["rts"]
                eic = inte["chrom"][0][9]["eic"]
                refRT = substances[substance]["RT"]
                
                ## generate replicates
                for repi in range(useEachInstanceNTimes):                        
                    ## add uniform Rt shift to EICs
                    artificialRTShift = 0
                    if repi > 0 and shiftRTs:
                        if inte["foundPeak"]:
                             ## shift according to peak boundaries
                            widthConstraint = 0.8 ## use entire chrom. peak width (=1) or less (0..1)
                            width = (inte["rtend"] - inte["rtstart"]) * widthConstraint
                            startRT = inte["rtstart"] + (1 - widthConstraint) / 2. * (inte["rtend"] - inte["rtstart"])
                            artificialRTShift = startRT + width * np.random.rand(1) - refRT
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
                    if repi > 0 and addRandomNoise:
                        ## randomize signal intensitiers
                        if np.random.rand(1)[0] > 0.5:
                            eicS = eicS * (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                        else:
                            eicS = eicS / (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                        
                        ## add noise on top of EIC
                        eicS = eicS + np.random.rand(eicS.shape[0]) * np.max(eicS) * maxNoiseLevelAdd
                    
                    ## test if eic has detected signals
                    if np.sum(eicS) > 0 and np.all(eicS >= 0):
                        ## add instance to training data
                        if temp is None:
                            temp = {"channel.rt"        : np.zeros((exportBatchSize, PeakBotMRM.Config.RTSLICES), dtype=float),
                                    "channel.int"       : np.zeros((exportBatchSize, PeakBotMRM.Config.RTSLICES), dtype=float),
                                    "inte.peak"         : np.zeros((exportBatchSize, PeakBotMRM.Config.NUMCLASSES), dtype=int),
                                    "inte.rtStart"      : np.zeros((exportBatchSize), dtype=float),
                                    "inte.rtEnd"        : np.zeros((exportBatchSize), dtype=float),
                                    "inte.rtInds"       : np.zeros((exportBatchSize, 2), dtype=float),
                                    "inte.area"         : np.zeros((exportBatchSize), dtype=float),
                                }
                            if includeMetaInfo:
                                temp = {**temp, 
                                        **{"ref.substanceName" : ["" for i in range(exportBatchSize)],
                                           "ref.sample"        : ["" for i in range(exportBatchSize)],
                                           "ref.rt"            : np.zeros((exportBatchSize), dtype=float),
                                           "ref.PeakForm"      : ["" for i in range(exportBatchSize)], 
                                           "ref.Rt shifts"     : ["" for i in range(exportBatchSize)],
                                           "ref.Note"          : ["" for i in range(exportBatchSize)],
                                           "loss.IOU_Area"     : np.ones((exportBatchSize), dtype=float),
                                        }
                                }
                                
                            curI = 0

                        assert curI < temp["channel.rt"].shape[0]
                        peakType = 0 if inte["foundPeak"] else 1

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

                        if includeMetaInfo:
                            ## substance data
                            temp["ref.substanceName"][curI] = substance
                            temp["ref.sample"       ][curI] = sample
                            temp["ref.rt"           ][curI] = substances[substance]["RT"]
                            temp["ref.PeakForm"     ][curI] = substances[substance]["PeakForm"] 
                            temp["ref.Rt shifts"    ][curI] = substances[substance]["Rt shifts"]
                            temp["ref.Note"         ][curI] = substances[substance]["Note"]
                        
                        curI = curI + 1
                    else:
                        np.set_printoptions(edgeitems=PeakBotMRM.Config.RTSLICES + 2, 
                            formatter=dict(float=lambda x: "%.3g" % x))
                        print(eicS)

                    ## if batch has been filled, export it to a temporary file
                    if curI >= exportBatchSize:
                        with open(os.path.join(instanceDir, "%s%d.pickle"%(PeakBotMRM.Config.INSTANCEPREFIX, cur)), "wb") as fout:
                            pickle.dump(temp, fout)
                            temp = None
                            curI = 0
                            cur += 1
    print("  | .. Exported batches each with %d instances (saved to '%s')."%(exportBatchSize, os.path.join(expDir, "trainingInstances.zip")))



def trainPeakBotMRMModel(expName, targetFile, curatedPeaks, samplesPath, modelFile, expDir = None, logDir = None, historyObject = None, removeHistoryObject = False,
                         MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                         allowedMZOffset = 0.05, balanceDataset = True,
                         addRandomNoise = True, maxRandFactor = 0.1, maxNoiseLevelAdd=0.1, shiftRTs = True, maxShift = 0.15, useEachInstanceNTimes = 5, 
                         excludeSubstances = None, includeSubstances = None, checkPeakAttributes = None, 
                         exportBatchSize = 1024,
                         comment="None"):
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
    histAll = None
    if removeHistoryObject:
        try: 
            os.remove(historyObject)
        except:
            pass
    else:
        try:
            histAll = pd.read_pickle(historyObject)
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


    substances               = PeakBotMRM.loadTargets(targetFile, excludeSubstances = excludeSubstances, includeSubstances = includeSubstances)
    substances, integrations = PeakBotMRM.loadIntegrations(substances, curatedPeaks)
    substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, samplesPath, expDir,
                                                             allowedMZOffset = allowedMZOffset, 
                                                             MRMHeader = MRMHeader)
        
        
    print("Peak statistics")
    tic()
    stats = {"hasPeak":0, "hasNoPeak":0, "peakProperties":[]}
    for substance in tqdm.tqdm(integrations.keys(), desc="  | .. calculating"):
        for sample in integrations[substance].keys():
            inte = integrations[substance][sample]
            if len(inte["chrom"]) == 1:
                rts = inte["chrom"][0][9]["rts"]
                eic = inte["chrom"][0][9]["eic"]
                refRT = substances[substance]["RT"]
                
                if inte["foundPeak"]:
                    stats["hasPeak"] = stats["hasPeak"] + 1
                    
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                    eicS[rtsS < inte["rtstart"]] = 0
                    eicS[rtsS > inte["rtend"]] = 0
                    apexRT = rtsS[np.argmax(eicS)]
                    
                    intLeft  = eicS[np.argmin(np.abs(rtsS - inte["rtstart"]))]
                    intRight = eicS[np.argmin(np.abs(rtsS - inte["rtend"]))]
                    intApex  = eicS[np.argmin(np.abs(rtsS - apexRT))]
                    
                    peakWidth = inte["rtend"] - inte["rtstart"]
                    centerOffset = apexRT - refRT
                    peakLeftInflection = inte["rtstart"] - apexRT
                    peakRightInflection = inte["rtend"] - apexRT
                    leftIntensityRatio = intApex/intLeft if intLeft > 0 else np.Inf
                    rightIntensityRatio = intApex/intRight if intRight > 0 else np.Inf                    
                    
                    stats["peakProperties"].append([sample, "width", peakWidth])
                    stats["peakProperties"].append([sample, "centerOffset", centerOffset])
                    stats["peakProperties"].append([sample, "leftInflection", peakLeftInflection])
                    stats["peakProperties"].append([sample, "rightInflection", peakRightInflection])
                    stats["peakProperties"].append([sample, "leftIntensityRatio", leftIntensityRatio])
                    stats["peakProperties"].append([sample, "rightIntensityRatio", rightIntensityRatio])
                    if intLeft > 0:
                        stats["peakProperties"].append([sample, "leftIntensityRatioNonInf", intApex/intLeft])
                    if intRight > 0:
                        stats["peakProperties"].append([sample, "rightIntensityRatioNonInf", intApex/intRight])
                    
                else:
                    stats["hasNoPeak"] = stats["hasNoPeak"] + 1
    df = pd.DataFrame(stats["peakProperties"], columns = ["sample", "type", "value"])
    print("  | .. There are %d peaks and %d Nopeaks. An overview of the peak stats has been saved to '%s'"%(stats["hasPeak"], stats["hasNoPeak"], os.path.join(expDir, "fig_peakStats.png")))
    print("  | .. .. The distribution of the peaks' properties (offset of apex to expected rt, left and right extends relative to peak apex, peak widths) are: (in minutes)")
    print(df.groupby("type").describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    df.drop(df[df.type == "leftIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "rightIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "leftIntensityRatioNonInf"].index, inplace=True)
    df.drop(df[df.type == "rightIntensityRatioNonInf"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free_y", ncol=2)
            + p9.ggtitle("Peak metrics") + p9.xlab("retention time") + p9.ylab("Count")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "fig_peakStats.png"), width=5.2, height=5, dpi=300)
    print("  | .. took %.1f seconds"%toc())
    print("\n")
    

    print("Balancing training dataset (and applying optional peak statistic filter criteria)")
    tic()
    peaks = []
    noPeaks = []
    notUsed = 0
    for substance in tqdm.tqdm(substances.values(), desc="  | .. balancing"):
        if substance["Name"] in integrations.keys():
            for sample in integrations[substance["Name"]].keys():
                if integrations[substance["Name"]][sample]["foundPeak"]:
                    inte = integrations[substance["Name"]][sample]
                    if len(inte["chrom"]) == 1:
                        rts = inte["chrom"][0][9]["rts"]
                        eic = inte["chrom"][0][9]["eic"]
                        refRT = substances[substance["Name"]]["RT"]
                        
                        if inte["foundPeak"]:
                            stats["hasPeak"] = stats["hasPeak"] + 1
                            
                            rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                            eicS[rtsS < inte["rtstart"]] = 0
                            eicS[rtsS > inte["rtend"]] = 0
                            apexRT = rtsS[np.argmax(eicS)]
                            
                            intLeft  = eicS[np.argmin(np.abs(rtsS - inte["rtstart"]))]
                            intRight = eicS[np.argmin(np.abs(rtsS - inte["rtend"]))]
                            intApex  = eicS[np.argmin(np.abs(rtsS - apexRT))]
                            
                            peakWidth = inte["rtend"] - inte["rtstart"]
                            centerOffset = apexRT - refRT
                            peakLeftInflection = inte["rtstart"] - apexRT
                            peakRightInflection = inte["rtend"] - apexRT
                            leftIntensityRatio = intApex/intLeft if intLeft > 0 else np.Inf
                            rightIntensityRatio = intApex/intRight if intRight > 0 else np.Inf
                            
                            if checkPeakAttributes is None or checkPeakAttributes(peakWidth, centerOffset, peakLeftInflection, peakRightInflection, leftIntensityRatio, rightIntensityRatio, eicS, rtsS):
                                peaks.append((substance["Name"], sample))
                            else:
                                notUsed = notUsed + 1
                else:
                    noPeaks.append((substance["Name"], sample))
    print("  | .. there are %d peaks and %d backgrounds in the dataset."%(len(peaks), len(noPeaks)))
    if checkPeakAttributes is not None:
        print("  | .. .. %d peaks were not used due to peak abnormalities according to the user-provided peak-quality function checkPeakAttributes."%(notUsed))
    random.shuffle(peaks)
    random.shuffle(noPeaks)
    a = min(len(peaks), len(noPeaks))
    if balanceDataset:
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
    if balanceDataset:
        print("  | .. balanced dataset to %d peaks and %d backgrounds"%(len(peaks), len(noPeaks)))
    else:
        print("  | .. dataset not balanced with %d peaks and %d backgrounds"%(len(peaks), len(noPeaks)))
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")


    print("Exporting original instances for validation")
    tic()
    instanceOriDirObj = tempfile.TemporaryDirectory(prefix="PBMRM_oriIns__")
    instanceOriDir = instanceOriDirObj.name
    compileInstanceDataset(expDir, substances, integrations, instanceOriDir, addRandomNoise = False, shiftRTs = False)
    ## randomize instances
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")


    print("Exporting augmented instances for training")
    tic()
    instanceDirObj = tempfile.TemporaryDirectory(prefix="PBMRM_all__")
    instanceDir = instanceDirObj.name
    compileInstanceDataset(expDir, substances, integrations, instanceDir, addRandomNoise = addRandomNoise, maxRandFactor = maxRandFactor, maxNoiseLevelAdd = maxNoiseLevelAdd, shiftRTs = shiftRTs, maxShift = maxShift, useEachInstanceNTimes = useEachInstanceNTimes)
    ## randomize instances
    PeakBotMRM.train.shuffleResultsSampleNames(instanceDir)
    PeakBotMRM.train.shuffleResults(instanceDir, steps=1E4, samplesToExchange=12)
    print("  | .. Instances shuffled")
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")


    ## train new model
    with tempfile.TemporaryDirectory(prefix="PBMRM_training__") as tempTrainDir, tempfile.TemporaryDirectory(prefix="PBMRM_validation__") as tempValDir:
        splitRatio = 0.7
        PeakBotMRM.train.splitDSinto(instanceDir, 
                                     newDS1Path = tempTrainDir, newDS2Path = tempValDir, 
                                     copy = True, ratioDS1 = splitRatio, verbose = False)
        
        nAllBatches     = len([f for f in os.listdir(instanceDir)    if os.path.isfile(os.path.join(instanceDir   , f))])
        nTrainBatches   = len([f for f in os.listdir(tempTrainDir)   if os.path.isfile(os.path.join(tempTrainDir  , f))])
        nValBatches     = len([f for f in os.listdir(tempValDir)     if os.path.isfile(os.path.join(tempValDir    , f))])
        nOriInstBatches = len([f for f in os.listdir(instanceOriDir) if os.path.isfile(os.path.join(instanceOriDir, f))])
        print("Split dataset")  
        tic()
        print("  | .. Randomly split dataset '%s' into a training and validation dataset with %.1f and %.1f parts of the instances "%(expDir, splitRatio, 1-splitRatio))
        print("  | .. There are %d training (%s) and %d validation (%s) batches available"%(nTrainBatches, tempTrainDir, nValBatches, tempValDir))
        print("  | .. With the current configuration (%d batchsize, %d steps per epoch) this will allow for %d epochs of training"%(PeakBotMRM.Config.BATCHSIZE, PeakBotMRM.Config.STEPSPEREPOCH, math.floor(nTrainBatches/PeakBotMRM.Config.STEPSPEREPOCH)))
        print("  | .. took %.1f seconds"%(toc()))
        print("\n")
        
        addValDS = []
        addValDS.append({"folder": instanceDir   , "name": "all" ,   "numBatches": nAllBatches -1})
        addValDS.append({"folder": tempTrainDir  , "name": "train" , "numBatches": nTrainBatches-1})
        addValDS.append({"folder": tempValDir    , "name": "val"   , "numBatches": nValBatches  -1})            
        addValDS.append({"folder": instanceOriDir, "name": "oriIns", "numBatches": nOriInstBatches-1})

        ## Train new peakbotMRM model
        pb, hist = PeakBotMRM.trainPeakBotMRMModel(trainInstancesPath = tempTrainDir,
                                                   addValidationInstances = addValDS,
                                                   logBaseDir = logDir,
                                                   everyNthEpoch = -1, 
                                                   verbose = True)

        pb.saveModelToFile(modelFile)
        print("Newly trained PeakBotMRM saved to file '%s'"%(modelFile))
        print("\n")

        ## add current history
        hist["comment"] = comment
        if histAll is None:
            histAll = hist
        else:
            histAll = histAll.append(hist, ignore_index=True)


        ### Summarize the training and validation metrices and losses
        histAll.to_pickle(historyObject)        

        ### Summarize and illustrate the results of the different training and validation dataset
        df = histAll
        df['ID'] = df.model.str.split('_').str[-1]
        df = df[df["metric"]!="loss"]
        plot = (p9.ggplot(df, p9.aes("set", "value", colour="set"))
                #+ p9.geom_violin()
                + p9.geom_jitter(height=0, alpha=0.5)
                + p9.facet_grid("metric~comment", scales="free_y")
                + p9.ggtitle("Training losses/metrics") + p9.xlab("Training/Validation dataset") + p9.ylab("Value")
                + p9.theme(legend_position = "none", axis_text_x=p9.element_text(angle=45)))
        p9.options.figure_size = (5.2, 7)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "fig_SummaryPlot.png"), width=5, height=8, dpi=300)
    print("\n\n\n")

from PeakBotMRM.core import tic, toc, arg_find_nearest, extractStandardizedEIC, getInteRTIndsOnStandardizedEIC
import PeakBotMRM

import PeakBotMRM.validate

from datetime import datetime
import time
import platform
import os
import math
import plotnine as p9
import pandas as pd
import numpy as np
import tqdm
import random
random.seed(2021)

import portalocker



def compileInstanceDataset(substances, integrations, experimentName, dataset = None, 
                           addRandomNoise=False, maxRandFactor=0.1, maxNoiseLevelAdd=0.1, 
                           shiftRTs=False, maxShift=0.1, useEachInstanceNTimes=1, balanceReps = False, 
                           aug_augment = True, aug_OnlyUseAugmentationFromSamples = None, aug_addAugPeaks = True, aug_maxAugPeaksN = 3, aug_plotAugInstances = False, 
                           verbose = True, logPrefix = ""):
    
    augPeaks = []
    augBackgrounds = []
    if aug_augment:
        ## extract peaks and backgrounds for synthetic dataset generation and for data augmentation (e.g., add peaks on top of existing eics)
        if verbose: 
            print(logPrefix, "  | .. Extracting peaks and backgrounds for augmentation")
        for substance in tqdm.tqdm(integrations.keys(), desc = logPrefix + "   | .. reference extraction"):
            for sample in integrations[substance].keys():
                key = "%s $ %s"%(substance, sample)
                inte = integrations[substance][sample]

                if inte.chromatogram is not None:
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]

                    if inte.foundPeak and (aug_OnlyUseAugmentationFromSamples is None or key in aug_OnlyUseAugmentationFromSamples):
                        startInd = np.argmin(np.abs(rts - inte.rtStart))
                        endInd = np.argmin(np.abs(rts - inte.rtEnd))
                        if endInd - startInd >= 2:
                            augPeaks.append({"rtstart"    : inte.rtStart,
                                            "rtend"      : inte.rtEnd,
                                            "eiccropped" : eic[startInd:endInd] - np.min(eic[startInd:endInd]),
                                            "rtscropped" : rts[startInd:endInd],
                                            "area"       : inte.area,
                                            "substance"  : substance,
                                            "sample"     : sample
                                            })

                    if not inte.foundPeak and (aug_OnlyUseAugmentationFromSamples is None or key in aug_OnlyUseAugmentationFromSamples):
                        augBackgrounds.append({"eic"       : eic,
                                               "rts"       : rts,
                                               "substance" : substance,
                                               "sample"    : sample
                                               })
    
    template = None    
    curInstanceInd = 0
    totalInstances = 0
    if dataset is None:
        dataset = PeakBotMRM.MemoryDataset()
        if dataset.data != None:
            totalInstances = dataset.data["channel.rt"].shape[0]
    elif verbose:
        print(logPrefix, "  | .. %d instances already present in the dataset. Appending..."%(dataset.data["channel.rt"].shape[0]))
    
    
    if addRandomNoise and verbose:
        print(logPrefix, "  | .. Random noise will be added. The range of the randomly generated factors is %.3f - %.3f and the maximum randomly-generated noise added on top of the EICs is %.3f"%(1/(1 + maxRandFactor), 1 + maxRandFactor, maxNoiseLevelAdd))
    if shiftRTs:
        print(logPrefix, "  | .. Random RT shifts will be added. The range is -%.3f - %.3f minutes"%(maxShift, maxShift))
        print(logPrefix, "  | .. Note: the first instance will be the unmodified, original EIC")
        print(logPrefix, "  | .. Chromatographic peaks with a shifted peak apex will first be corrected to the designated RT and then randomly moved for the training instance")
    if aug_augment: 
        print(logPrefix, "  | .. Instances will be augmented")
        if aug_addAugPeaks: 
            print(logPrefix, "  | .. .. with chromatographic peaks (maximum of %d). for more details please refer to the code"%(aug_maxAugPeaksN))
    if verbose: 
        print(logPrefix, "  | .. Each instance shall be used %d times and the peak/background classes shall%s be balanced"%(useEachInstanceNTimes, "" if balanceReps else " not"))
    
    useEachPeakInstanceNTimes = useEachInstanceNTimes
    useEachBackgroundInstanceNTimes = useEachInstanceNTimes
    if balanceReps:
        peaks = 0
        noPeaks = 0
        for substance in integrations.keys():
            for sample in integrations[substance].keys():
                inte = integrations[substance][sample]
                if inte.chromatogram is not None:
                    if inte.foundPeak:
                        peaks += 1
                    else:
                        noPeaks += 1
        useEachPeakInstanceNTimes = int(round(useEachInstanceNTimes / (peaks / max(peaks, noPeaks))))
        useEachBackgroundInstanceNTimes = int(round(useEachInstanceNTimes / (noPeaks / max(peaks, noPeaks))))
    if verbose:
        print(logPrefix, "  | .. Each peak instance will be used %d times and each background instance %d times"%(useEachPeakInstanceNTimes, useEachBackgroundInstanceNTimes))
    insNum = 0
    for substanceName in tqdm.tqdm(integrations.keys(), desc=logPrefix + "   | .. compiling substance"):
        refRT = substances[substanceName].refRT
        for sample in integrations[substanceName].keys():
            inte = integrations[substanceName][sample]
            if inte.chromatogram is not None:
                rts = inte.chromatogram["rts"]
                eic = inte.chromatogram["eic"]
                
                ## generate replicates
                reps = useEachPeakInstanceNTimes if inte.foundPeak else useEachBackgroundInstanceNTimes
                for repi in range(reps):
                    ## add uniform Rt shift to EICs
                    artificialRTShift = 0
                    if repi > 0 and shiftRTs:
                        if inte.foundPeak:
                             ## shift according to peak boundaries
                            widthConstraint = 0.8 ## use entire chrom. peak width (=1) or less (0..1)
                            width = (inte.rtEnd - inte.rtStart) * widthConstraint
                            startRT = inte.rtStart + (1 - widthConstraint) / 2. * (inte.rtEnd - inte.rtStart)
                            artificialRTShift = startRT + width * np.random.rand(1) - refRT
                        else:
                            artificialRTShift = np.random.rand(1) * 2 * maxShift - maxShift
                    
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT + artificialRTShift)
                    
                    ## get integration results on standardized area
                    bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                        getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                       inte.foundPeak, 
                                                       inte.rtStart, 
                                                       inte.rtEnd)
                    
                    ## add random noise
                    if repi > 0 and addRandomNoise:
                        ## randomize signal intensitiers
                        if np.random.rand(1)[0] > 0.5:
                            eicS = eicS * (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                        else:
                            eicS = eicS / (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                        
                        ## add noise on top of EIC
                        eicS = eicS + np.ones(eicS.shape[0]) * np.random.rand(1)[0] * np.max(eicS) * maxNoiseLevelAdd
                        
                    ## randomly add augmentation peak
                    if repi > 0 and aug_augment and aug_addAugPeaks: 
                        ## random number of augmentation peaks to be added
                        augPeaksToAdd = np.random.randint(1, aug_maxAugPeaksN)
                        
                        ## initialize empty EIC
                        peakSignalType = np.zeros(PeakBotMRM.Config.RTSLICES, dtype=int)
                        if inte.foundPeak:
                            peakSignalType[bestRTStartInd:bestRTEndInd] = 1
                        
                        dat = {"eic": [], "rts": [], "type": []}
                        if aug_plotAugInstances:
                            for i in range(eicS.shape[0]):
                                if(rtsS[i] > 0):
                                    dat["eic"].append(eicS[i])
                                    dat["rts"].append(rtsS[i])
                                    dat["type"].append("ori")
                            
                        peaksPop = 1
                        maxTries = 10
                        while augPeaksToAdd > 0 and maxTries > 0:
                            maxTries = maxTries - 1
                            
                            ## randomly select a peak to be used and a position to add peak at
                            pI = np.random.randint(0, len(augPeaks))
                            peakEIC = augPeaks[pI]["eiccropped"]
                            addAtInd = np.random.randint(math.ceil(PeakBotMRM.Config.RTSLICES*0.1), math.floor(PeakBotMRM.Config.RTSLICES*0.9))
                            addstartEIC = max(addAtInd - math.floor(peakEIC.shape[0]/2), 0)
                            addendEIC = min(addAtInd + math.ceil(peakEIC.shape[0]/2), PeakBotMRM.Config.RTSLICES)
                            peakUseStart = max(math.floor(peakEIC.shape[0]/2) - addAtInd, 0)
                            peakUseEnd = min(peakEIC.shape[0], PeakBotMRM.Config.RTSLICES - addAtInd + math.floor(peakEIC.shape[0]/2))

                            ## try and add peak at particular position                            
                            scaling = np.random.random() * 2 + 1 if random.randint(0,1) == 1 else 1 / (np.random.random() * 2 + 1)
                            eicSTemp = np.copy(eicS)
                            peakSignalTypeTemp = np.copy(peakSignalType)
                            eicSTemp[addstartEIC:addendEIC] = eicSTemp[addstartEIC:addendEIC] + peakEIC[peakUseStart:peakUseEnd] / np.max(peakEIC[peakUseStart:peakUseEnd]) * np.max(eicSTemp) * scaling
                            peakSignalTypeTemp[addstartEIC:addendEIC] = peakSignalTypeTemp[addstartEIC:addendEIC] + 2**peaksPop

                            ## if there are overlapping peaks diretly in the eics center, discard the addition
                            if inte.foundPeak and any(peakSignalTypeTemp[bestRTStartInd : bestRTEndInd] > 1):
                                continue
                            elif not inte.foundPeak and any(peakSignalTypeTemp[math.ceil(PeakBotMRM.Config.RTSLICES*0.45) : math.floor(PeakBotMRM.Config.RTSLICES*0.55)] > 1):
                                continue

                            eicS = eicSTemp
                            peakSignalType = peakSignalTypeTemp

                            augPeaksToAdd = augPeaksToAdd - 1
                            peaksPop += 1
                        
                        if aug_plotAugInstances:
                            for i in range(eicS.shape[0]):
                                if(rtsS[i] > 0):
                                    dat["eic"].append(eicS[i])
                                    dat["rts"].append(rtsS[i])
                                    dat["type"].append("aug")
                            
                            plot = (p9.ggplot(pd.DataFrame(dat), p9.aes(x='rts', y='eic', colour="type"))
                                + p9.geom_line()
                                + p9.ggtitle("Instance with %s"%("peak" if inte.foundPeak else "background"))
                            )
                            if inte.foundPeak:
                                plot = plot + p9.geom_vline(xintercept = [inte.rtStart, inte.rtEnd])
                            else:
                                plot = plot + p9.geom_vline(xintercept = [refRT])
                            p9.ggsave(plot=plot, filename=os.path.join("./Training", "%d.png"%(insNum)), width=7, height=4, dpi=72, limitsize=False, verbose=False)
                        insNum += 1
                                                
                    ## test if eic has detected signals
                    if np.sum(eicS) > 0 and np.all(eicS >= 0):
                        ## add instance to training data
                        if template is None or curInstanceInd >= template["channel.rt"].shape[0]:
                            template = PeakBotMRM.getDatasetTemplate()
                            curInstanceInd = 0

                        ## analytical raw data
                        template["channel.rt"       ][curInstanceInd,:] = rtsS
                        template["channel.int"      ][curInstanceInd,:] = eicS

                        ## manual integration data
                        template["inte.peak"        ][curInstanceInd, 0 if inte.foundPeak else 1] = 1
                        template["inte.rtStart"     ][curInstanceInd]   = bestRTStart
                        template["inte.rtEnd"       ][curInstanceInd]   = bestRTEnd
                        template["inte.rtInds"      ][curInstanceInd,0] = bestRTStartInd
                        template["inte.rtInds"      ][curInstanceInd,1] = bestRTEndInd                        
                        template["inte.area"        ][curInstanceInd]   = inte.area
                        template["pred"             ][curInstanceInd]   = np.hstack((template["inte.peak"][curInstanceInd,:], template["inte.rtInds"][curInstanceInd,:], template["channel.int"][curInstanceInd,:]))

                        if PeakBotMRM.Config.INCLUDEMETAINFORMATION:
                            ## substance data
                            template["ref.substance" ][curInstanceInd] = substanceName
                            template["ref.sample"    ][curInstanceInd] = sample
                            template["ref.experiment"][curInstanceInd] = experimentName + ";" + sample + ";" + substanceName
                            template["ref.rt"        ][curInstanceInd] = substances[substanceName].refRT
                            template["ref.PeakForm"  ][curInstanceInd] = substances[substanceName].peakForm
                            template["ref.Rt shifts" ][curInstanceInd] = substances[substanceName].rtShift
                            template["ref.Note"      ][curInstanceInd] = substances[substanceName].note
                            template["ref.criteria"  ][curInstanceInd] = substances[substanceName].criteria
                            template["ref.polarity"  ][curInstanceInd] = substances[substanceName].polarity
                            template["ref.type"      ][curInstanceInd] = substances[substanceName].type
                            template["ref.CE"        ][curInstanceInd] = substances[substanceName].CE
                            template["ref.CMethod"   ][curInstanceInd] = substances[substanceName].CEMethod
                            template["loss.IOU_Area" ][curInstanceInd] = 1
                        
                        curInstanceInd += 1
                        totalInstances += 1
                    else:
                        np.set_printoptions(edgeitems=PeakBotMRM.Config.RTSLICES + 2, formatter=dict(float=lambda x: "%.3g" % x))
                        print(eicS)
                        raise RuntimeError("Problem with EIC")

                    ## if batch has been filled, export it to a temporary file
                    if curInstanceInd >= template["channel.rt"].shape[0]:
                        dataset.addData(template)
                        template = None
                        curInstanceInd = 0
    dataset.addData(template)
    if dataset.getElements() > 0:
        dataset.removeOtherThan(0, totalInstances)
    if verbose:
        print(logPrefix, "  | .. Exported %d instances."%(dataset.getElements()))
        
    return dataset

def generateAndExportAugmentedInstancesForTraining(substances, integrations, experimentName, 
                                                   addRandomNoise, maxRandFactor, maxNoiseLevelAdd, 
                                                   shiftRTs, maxShift, useEachInstanceNTimes, balanceAugmentations, 
                                                   aug_augment, aug_OnlyUseAugmentationFromSamples, aug_addAugPeaks, aug_maxAugPeaksN, aug_plotAugInstances, 
                                                   dataset = None, verbose = True, logPrefix = ""):
    if verbose: 
        print(logPrefix, "Exporting augmented instances for training")
    tic()
    dataset = compileInstanceDataset(substances, integrations, experimentName, dataset = dataset, 
                                     addRandomNoise = addRandomNoise, maxRandFactor = maxRandFactor, maxNoiseLevelAdd = maxNoiseLevelAdd, 
                                     shiftRTs = shiftRTs, maxShift = maxShift, 
                                     useEachInstanceNTimes = useEachInstanceNTimes, balanceReps = balanceAugmentations, 
                                     aug_augment = aug_augment, aug_OnlyUseAugmentationFromSamples = aug_OnlyUseAugmentationFromSamples, aug_addAugPeaks = aug_addAugPeaks, aug_maxAugPeaksN = aug_maxAugPeaksN, aug_plotAugInstances = aug_plotAugInstances, 
                                     verbose = verbose, logPrefix = logPrefix)
    if verbose: 
        print(logPrefix, "  | .. took %.1f seconds"%(toc()))
        print(logPrefix)
    return dataset

def exportOriginalInstancesForValidation(substances, integrations, experimentName, dataset = None, verbose = True, logPrefix = ""):
    if verbose:
        print(logPrefix, "Exporting original instances for validation")
    tic()
    dataset = compileInstanceDataset(substances, integrations, experimentName, dataset = dataset, addRandomNoise = False, shiftRTs = False, aug_augment = False, verbose = verbose, logPrefix = logPrefix)
    if verbose: 
        print(logPrefix, "  | .. took %.1f seconds"%(toc()))
        print(logPrefix)
    return dataset

def constrainAndBalanceDataset(balanceDataset, checkPeakAttributes, substances, integrations, verbose = True, logPrefix = ""):
    if verbose: 
        print(logPrefix, "Inspecting dataset (and applying optional peak statistic filter criteria)")
    assert checkPeakAttributes is None or checkPeakAttributes[0] == 0 or (checkPeakAttributes[0] != 0 and checkPeakAttributes[1] is not None)
    tic()
    peaks = []
    noPeaks = []
    useCriteria = {}
    notUsedCount = 0
    for substance in tqdm.tqdm(substances.values(), desc=logPrefix + "   | .. inspecting"):
        if substance.name in integrations.keys():
            for sample in integrations[substance.name].keys():
                if integrations[substance.name][sample].foundPeak:
                    inte = integrations[substance.name][sample]
                    if inte.chromatogram is not None:
                        rts = inte.chromatogram["rts"]
                        eic = inte.chromatogram["eic"]
                        refRT = substance.refRT
                        rtStart = inte.rtStart
                        rtEnd = inte.rtEnd
                        
                        rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                        eicS[rtsS < rtStart] = 0
                        eicS[rtsS > rtEnd] = 0
                        apexRT = rtsS[np.argmax(eicS)]
                        
                        intLeft  = eicS[arg_find_nearest(rtsS, rtStart)]
                        intRight = eicS[arg_find_nearest(rtsS, rtEnd)]
                        intApex  = eicS[arg_find_nearest(rtsS, apexRT)]
                        
                        peakWidth = rtEnd - rtStart
                        centerOffset = apexRT - refRT
                        peakLeftInflection = intLeft - apexRT
                        peakRightInflection = intRight - apexRT
                        leftIntensityRatio = intApex/intLeft if intLeft > 0 else np.Inf
                        rightIntensityRatio = intApex/intRight if intRight > 0 else np.Inf
                        
                        use = True, "check disabled"
                        if checkPeakAttributes is not None:
                            use = True, "no check"
                            if checkPeakAttributes[0] != 0:
                                use = checkPeakAttributes[1](peakWidth, centerOffset, peakLeftInflection, peakRightInflection, leftIntensityRatio, rightIntensityRatio, eicS, rtsS)
                            use = (checkPeakAttributes[0] == 0 or (checkPeakAttributes[0] == -1 and not use[0]) or (checkPeakAttributes[0] == 1 and use[0]), use[1])
                            
                        if use[0]:
                            peaks.append((substance.name, sample))
                        else:
                            notUsedCount += 1
                        if use[1] not in useCriteria.keys():
                            useCriteria[use[1]] = 0
                        useCriteria[use[1]] += 1
                        
                else:
                    noPeaks.append((substance.name, sample))
    if checkPeakAttributes is not None and verbose:
        for k, v in useCriteria.items():
            print(logPrefix, "  | .. .. %d (%.1f%%) peaks used / not used due to '%s'"%(v, v/(notUsedCount+len(peaks))*100, k))
        
        print(logPrefix, "  | .. .. %d (%.1f%%) of %d peaks were not used due to peak abnormalities according to the user-provided peak-quality function checkPeakAttributes."%(notUsedCount, notUsedCount/(notUsedCount + len(peaks))*100 if (notUsedCount + len(peaks)) > 0 else 0, notUsedCount + len(peaks)))
        
    if verbose: 
        print(logPrefix, "  | .. there are (after peak attribute checking) %d (%.1f%%) peaks and %d (%.1f%%) backgrounds in the dataset."%(len(peaks), len(peaks)/(len(peaks) + len(noPeaks))*100 if len(peaks) + len(noPeaks) > 0 else 0, len(noPeaks), len(noPeaks)/(len(peaks) + len(noPeaks))*100 if (len(peaks) + len(noPeaks)) > 0 else 0))
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
        if substance.name in integrations.keys():
            for sample in integrations[substance.name].keys():
                if integrations[substance.name][sample].foundPeak:
                    peaks.append((substance.name, sample))
                else:
                    noPeaks.append((substance.name, sample))
    if verbose:
        if balanceDataset:
            print(logPrefix, "  | .. balanced dataset randomly to %d peaks and %d backgrounds"%(len(peaks), len(noPeaks)))
        print(logPrefix, "  | .. took %.1f seconds"%(toc()))
        print(logPrefix)
    return integrations

def investigatePeakMetrics(expDir, substances, integrations, expName = "", plot = True, print2Console = True, verbose = True, logPrefix = ""):
    if verbose:
        print(logPrefix, "Peak statistics for '%s'"%(expName))
    tic()
    stats = {"hasPeak":0, "hasNoPeak":0, "peakProperties":[]}
    for substanceName in tqdm.tqdm(integrations.keys(), desc="   | .. calculating"):
        for sample in integrations[substanceName].keys():
            inte = integrations[substanceName][sample]
            if substanceName in substances.keys() and inte.chromatogram is not None:
                rts = inte.chromatogram["rts"]
                eic = inte.chromatogram["eic"]
                refRT = substances[substanceName].refRT        
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                
                sampleTyp = "sample"
                if "_BLK" in sample:
                    sampleTyp = "Blank"
                elif "_CAL" in sample:
                    sampleTyp = "Calibration"
                            
                if inte.foundPeak:
                    stats["hasPeak"] = stats["hasPeak"] + 1
                    
                    eicS[rtsS < inte.rtStart] = 0
                    eicS[rtsS > inte.rtEnd] = 0
                    apexRT = rtsS[np.argmax(eicS)]
                    
                    intLeft  = eicS[np.argmin(np.abs(rtsS - inte.rtStart))]
                    intRight = eicS[np.argmin(np.abs(rtsS - inte.rtEnd))]
                    intApex  = eicS[np.argmin(np.abs(rtsS - apexRT))]
                    
                    peakWidth = inte.rtEnd - inte.rtStart
                    peakWidthScans = np.argmin(np.abs(rtsS - inte.rtEnd)) - np.argmin(np.abs(rtsS - inte.rtStart))
                    centerOffset = apexRT - refRT
                    peakLeftInflection = inte.rtStart - apexRT
                    peakRightInflection = inte.rtEnd - apexRT
                    leftIntensityRatio = intApex/intLeft if intLeft > 0 else np.Inf
                    rightIntensityRatio = intApex/intRight if intRight > 0 else np.Inf
                    
                    refRTPos = 0
                    if refRT < inte.rtStart:
                        refRTPos = 1
                    elif refRT < apexRT:
                        refRTPos = (apexRT - refRT) / (apexRT - inte.rtStart)
                    elif inte.rtEnd < refRT:
                        refRTPos = -1
                    else: # apexRT < refRT
                        refRTPos = - (refRT - apexRT) / (inte.rtEnd - apexRT)
                        
                    area = inte.area
                    areaPB = PeakBotMRM.integrateArea(eic, rts, inte.rtStart, inte.rtEnd)
                    areaEIC = PeakBotMRM.integrateArea(eic, rts, -1, 1E4)
                    
                    diff = eicS[np.logical_and(rtsS >= inte.rtStart, rtsS <= inte.rtEnd)]
                    diff = np.diff(diff)
                    diff = diff / np.abs(diff)
                    nonSmooth = np.sum(diff[:(diff.shape[0]-1)] != diff[1:]) / diff.shape[0]
                    
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "nonSmoothVal", nonSmooth])    
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakArea", area])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakAreaPB", areaPB])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakAreaPBExplained", areaPB/areaEIC])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakWidth", peakWidth])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakWidthScans", peakWidthScans])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "apexReferenceOffset", centerOffset])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakLeftInflection", peakLeftInflection])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakRightInflection", peakRightInflection])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakBorderLeftIntensityRatio", leftIntensityRatio])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakBorderRightIntensityRatio", rightIntensityRatio])
                    if intLeft > 0:
                        stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakBorderLeftIntensityRatioNonInf", intApex/intLeft])
                    if intRight > 0:
                        stats["peakProperties"].append([sample, sampleTyp, substanceName, "peakBorderRightIntensityRatioNonInf", intApex/intRight])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "eicStandStartToRef", min(rtsS[rtsS>0]) - refRT])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "eicStandEndToRef", max(rtsS) - refRT])
                    stats["peakProperties"].append([sample, sampleTyp, substanceName, "refRTPos", refRTPos])
                    
                else:
                    stats["hasNoPeak"] = stats["hasNoPeak"] + 1
                
    tf = pd.DataFrame(stats["peakProperties"], columns = ["sample", "sampletype", "substance", "type", "value"])
    tf.insert(0, "Experiment", [expName for i in range(tf.shape[0])])
    if print2Console:
        print(logPrefix, "  | .. There are %d peaks and %d nopeaks. An overview of the peak stats has been saved to '%s'"%(stats["hasPeak"], stats["hasNoPeak"], os.path.join(expDir, "%s_peakStat.png"%(expName))))
        print(logPrefix, "  | .. .. The distribution of the peaks' properties (offset of apex to expected rt, left and right extends relative to peak apex, peak widths) are: (in minutes)")
        print(logPrefix, tf.groupby("type").describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    
    if plot:
        plotPeakMetrics(tf, expDir, expName)
    
    if verbose:
        if plot:
            print(logPrefix, "  | .. plotted peak metrics to file '%s'"%(os.path.join(expDir, "%s_peakStat.png"%(expName))))
        print(logPrefix, "  | .. took %.1f seconds"%toc())
        print(logPrefix)
        
    return tf

def plotPeakMetrics(tf, expDir, expName):
    df = tf.copy()
    df.drop(df[df.type == "peakWidthScans"].index, inplace=True)
    df.drop(df[df.type == "peakBorderLeftIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "peakBorderRightIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "peakBorderLeftIntensityRatioNonInf"].index, inplace=True)
    df.drop(df[df.type == "peakBorderRightIntensityRatioNonInf"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free_y", ncol=2)
            + p9.ggtitle("Peak metrics") + p9.xlab("retention time (minutes)") + p9.ylab("Frequency")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_1.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
    
    df = tf.copy()
    df.drop(df[df.type != "peakWidthScans"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free", ncol=2)
            + p9.ggtitle("Peak metrics") + p9.xlab("peak width (scans)") + p9.ylab("Frequency")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_2.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
    
    df = tf.copy()
    df.drop(df[df.type == "apexReferenceOffset"].index, inplace=True)
    df.drop(df[df.type == "eicStandEndToRef"].index, inplace=True)
    df.drop(df[df.type == "eicStandStartToRef"].index, inplace=True)
    df.drop(df[df.type == "peakLeftInflection"].index, inplace=True)
    df.drop(df[df.type == "peakRightInflection"].index, inplace=True)
    df.drop(df[df.type == "peakWidth"].index, inplace=True)
    df.drop(df[df.type == "peakWidthScans"].index, inplace=True)
    df.drop(df[df.type == "peakBorderLeftIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "peakBorderRightIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "refRTPos"].index, inplace=True)
    
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free", ncol=2)
            + p9.scales.scale_x_log10() + p9.scales.scale_y_log10()
            + p9.ggtitle("Peak metrics") + p9.xlab("-") + p9.ylab("Frequency")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_3.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
    
    df = tf.copy()
    df.drop(df[df.type != "peakWidth"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value", "substance", colour="sampletype"))
            + p9.geom_jitter(width=0, height=0.2, alpha=0.1)
            #+ p9.facet_wrap("~type", scales="free", ncol=2)
            #+ p9.scales.scale_x_log10() + p9.scales.scale_y_log10()
            + p9.scales.scale_y_discrete(limits=list(df.groupby(["substance"]).mean().sort_values(["value"], ascending = False).reset_index()["substance"]))
            + p9.ggtitle("Peak width per substance") + p9.xlab("Peak width (seconds)") + p9.ylab("Substance")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_4.png"%(expName)), width=5.2, height=35, dpi=300, limitsize=False, verbose=False)

def createHistory(histObjectFile, locationAndPrefix, verbose = True, logPrefix = ""):
    
    histAll = None
    with portalocker.Lock(histObjectFile, mode = "rb+", timeout = 60, check_interval = 2) as fh:
        histAll = pd.read_pickle(fh)
    
    temp = histAll[[i in ["Sensitivity (peaks)", "Specificity (no peaks)", "Area IOU"] for i in histAll.metric]]
    temp = temp.pivot(index = ["set", "model", "comment"], columns = "metric", values = "value").groupby(["set", "comment"]).describe()
    temp.to_csv(locationAndPrefix + "_resultsTable.tsv", sep="\t")
    
    df = histAll
    df = df[df["metric"] == "Area IOU"]
    print(logPrefix, "  | .. Plot for all instances simultaneously")
    plot = (p9.ggplot(df, p9.aes("value", "comment", group="comment"))
            + p9.geom_jitter(width = 0, height = 0.2, alpha=0.5)
            + p9.geom_vline(xintercept = 0.9)
            + p9.theme(legend_position = "bottom")
            + p9.theme_minimal()
            + p9.theme(plot_background = p9.element_rect(fill = "White"))
            + p9.ggtitle("Area IOU") + p9.xlab("Set") + p9.ylab("Area IOU") 
           )
    p9.options.figure_size = (5.2, 7)
    p9.ggsave(plot=plot, filename="%s_AreaIOU_models.png"%(locationAndPrefix), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    
    df = histAll
    df = df[df["metric"] == "Area IOU"]
    print(logPrefix, "  | .. Plot for all instances simultaneously")
    plot = (p9.ggplot(df, p9.aes("value", "set", group="set"))
            + p9.geom_jitter(width = 0, height = 0.2, alpha=0.5)
            + p9.geom_vline(xintercept = 0.9)
            + p9.theme(legend_position = "bottom")
            + p9.theme_minimal()
            + p9.theme(plot_background = p9.element_rect(fill = "White"))
            + p9.ggtitle("Area IOU") + p9.xlab("Set") + p9.ylab("Area IOU") 
           )
    p9.options.figure_size = (5.2, 7)
    p9.ggsave(plot=plot, filename="%s_AreaIOU_sets.png"%(locationAndPrefix), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    
    df = histAll
    df = df[df["metric"] == "Area IOU"]
    print(logPrefix, "  | .. Plot for all instances simultaneously")
    plot = (p9.ggplot(df, p9.aes("value", "set"))
            + p9.geom_jitter(width = 0, height = 0.2, alpha=0.5)
            + p9.geom_vline(xintercept = 0.9)
            + p9.facet_wrap("~model")
            + p9.theme(legend_position = "bottom")
            + p9.theme_minimal()
            + p9.theme(plot_background = p9.element_rect(fill = "White"))
            + p9.ggtitle("Area IOU") + p9.xlab("Set") + p9.ylab("Area IOU") 
           )
    p9.options.figure_size = (5.2, 7)
    p9.ggsave(plot=plot, filename="%s_AreaIOU.png"%(locationAndPrefix), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    
        
    ### Summarize and illustrate the results of the different training and validation dataset
    df = histAll
    print(df)
    df['ID'] = df.model.str.split('_').str[-1]
    df = df[df["metric"]!="loss"]
    df = df[[i in ["Sensitivity (peaks)", "Specificity (no peaks)"] for i in df.metric]]
    df.value  = df.apply(lambda x: x.value if x.metric != "Specificity (no peaks)" else 1 - x.value, axis=1)
    df.metric = df.apply(lambda x: "FPR" if x.metric == "Specificity (no peaks)" else x.metric, axis=1)
    df.metric = df.apply(lambda x: "TPR" if x.metric == "Sensitivity (peaks)" else x.metric, axis=1)
    df = df.pivot(index=["model", "set", "comment"], columns="metric", values="value")
    df.reset_index(inplace=True)
    
    print(logPrefix, "  | .. Plot for all instances simultaneously")
    plot = (p9.ggplot(df, p9.aes("FPR", "TPR", colour="comment", group="model"))
            + p9.geom_point(alpha=0.5)
            + p9.facet_wrap("~comment", ncol=4)
            + p9.theme(legend_position = "none")
            + p9.theme_minimal()
            + p9.theme(plot_background = p9.element_rect(fill = "White"))
            + p9.ggtitle("ROC") + p9.xlab("FPR") + p9.ylab("TPR") 
           )
    p9.options.figure_size = (5.2, 7)
    p9.ggsave(plot=plot, filename="%s_ROC.png"%(locationAndPrefix), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
    p9.ggsave(plot=p, filename="%s_ROC_zoomed.png"%(locationAndPrefix), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    
    if False:
        print(logPrefix, "  | .. Plot for all instances simultaneously")
        plot = (p9.ggplot(df, p9.aes("FPR", "TPR", colour="comment", group="model"))
                + p9.geom_point(alpha=0.5)
                + p9.geom_point(data = df[df["set"] == "Ref_R100140_AddVal_Ori"], size=4)
                + p9.facet_wrap("~model", ncol=6)
                + p9.theme(legend_position = "none")
                + p9.theme_minimal()
                + p9.theme(plot_background = p9.element_rect(fill = "White"))
                + p9.ggtitle("ROC") + p9.xlab("FPR") + p9.ylab("TPR") 
            )
        p9.options.figure_size = (5.2, 7)
        p9.ggsave(plot=plot, filename="%s_models_ROC.png"%(locationAndPrefix), width=20, height=15, dpi=300, limitsize=False, verbose=False)
        p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
        p9.ggsave(plot=p, filename="%s_models_ROC_zoomed.png"%(locationAndPrefix), width=20, height=15, dpi=300, limitsize=False, verbose=False)
        
        print(logPrefix, "  | .. Plot for all instances simultaneously")
        plot = (p9.ggplot(df, p9.aes("FPR", "TPR", colour="comment", group="model"))
                + p9.geom_point(alpha=0.5)
                + p9.geom_point(data = df[df["set"] == "Ref_R100140_AddVal_Ori"], size=4)
                + p9.facet_grid("comment~model")
                + p9.theme(legend_position = "none")
                + p9.theme_minimal()
                + p9.theme(plot_background = p9.element_rect(fill = "White"))
                + p9.ggtitle("ROC") + p9.xlab("FPR") + p9.ylab("TPR") 
            )
        p9.options.figure_size = (5.2, 7)
        p9.ggsave(plot=plot, filename="%s_models2_ROC.png"%(locationAndPrefix), width=60, height=15, dpi=300, limitsize=False, verbose=False)
        p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
        p9.ggsave(plot=p, filename="%s_models2_ROC_zoomed.png"%(locationAndPrefix), width=60, height=15, dpi=300, limitsize=False, verbose=False)
        
        print(logPrefix, "  | .. Plot for all instances simultaneously")
        temp = df[df["comment"] == " BalRan (8 ins)"]
        plot = (p9.ggplot(temp, p9.aes("FPR", "TPR", colour="comment", group="model"))
                + p9.geom_point(alpha=0.5)
                + p9.geom_point(data = temp[temp["set"] == "Ref_R100140_AddVal_Ori"], size=4)
                + p9.facet_wrap("~model", ncol=6)
                + p9.theme(legend_position = "none")
                + p9.theme_minimal()
                + p9.theme(plot_background = p9.element_rect(fill = "White"))
                + p9.ggtitle("ROC") + p9.xlab("FPR") + p9.ylab("TPR") 
            )
        p9.options.figure_size = (5.2, 7)
        p9.ggsave(plot=plot, filename="%s_models3_ROC.png"%(locationAndPrefix), width=20, height=15, dpi=300, limitsize=False, verbose=False)
        p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
        p9.ggsave(plot=p, filename="%s_models3_ROC_zoomed.png"%(locationAndPrefix), width=20, height=15, dpi=300, limitsize=False, verbose=False)
    
    for setName in set(df["set"]):
        try:
            print(logPrefix, "  | .. Plotting for set '%s'"%(setName))        
            temp = df[df["set"] == setName]
            
            plot = (p9.ggplot(temp, p9.aes("FPR", "TPR", colour="comment",  group="model"))
                    + p9.geom_point(alpha=0.5)
                    + p9.facet_wrap("~comment", ncol=4)
                    + p9.theme(legend_position = "none")
                    + p9.theme_minimal()
                    + p9.theme(plot_background = p9.element_rect(fill = "White"))
                    + p9.ggtitle("ROC for sets containing the string '%s'"%(setName)) + p9.xlab("FPR") + p9.ylab("TPR") 
                )
            p9.options.figure_size = (5.2, 7)
            p9.ggsave(plot=plot, filename="%s_ROC_%s.png"%(locationAndPrefix, setName), width=10, height=10, dpi=150, limitsize=False, verbose=False)
            p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
            p9.ggsave(plot=p, filename="%s_ROC_zoomed_%s.png"%(locationAndPrefix, setName), width=10, height=10, dpi=150, limitsize=False, verbose=False)
        except: 
            pass

def trainPeakBotMRMModel(expName, trainDSs, valDSs, modelFile, expDir = None, logDir = None, historyFile = None, 
                         MRMHeader = None,
                         allowedMZOffset = 0.05, balanceDataset = False, balanceAugmentations = True,
                         addRandomNoise = True, maxRandFactor = 0.1, maxNoiseLevelAdd=0.1, shiftRTs = True, maxShift = 0.15, useEachInstanceNTimes = 5, 
                         aug_augment = True, aug_OnlyUseAugmentationFromSamples = None, aug_addAugPeaks = True, aug_maxAugPeaksN = 3, aug_plotAugInstances = False, 
                         showPeakMetrics = True, 
                         stratifyDataset = True, intThres = 1E3, peakWidthThres = None,
                         comment="None", useDSForTraining = "augmented", 
                         verbose = True, logPrefix = ""):
    tic("Overall process")
    
    if MRMHeader is None:
        MRMHeader = PeakBotMRM.Config.MRMHEADER
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if historyFile is None:
        historyFile = os.path.join(expDir, "History.pandas.pickle")
    
    try:
        if not os.path.isdir(expDir):
            os.mkdir(expDir)
    except:
        print("Could not generate experiment directory '%s'"%(expDir))
        raise
    try:
        if not os.path.isdir(os.path.join(expDir, "SubstanceFigures")):
            os.mkdir(os.path.join(expDir, "SubstanceFigures"))
    except:
        print("Could not generate substance figure directory '%s'"%(os.path.join(expDir, "SubstanceFigures")))   
        raise
        
        
    print("Training model from experiment")
    print("  | .. Parameters")
    print("  | .. .. expName: '%s'"%(expName))
    print("  | .. .. modelFile: '%s'"%(modelFile))
    print("  | .. .. expDir: '%s'"%(expDir))
    print("  | .. .. logDir: '%s'"%(logDir))
    print("  | .. .. MRMHeader: '%s'"%(MRMHeader))
    print("  | .. .. allowedMZOffset: '%s'"%(allowedMZOffset))
    print("  | .. .. addRandomNoise: '%s'"%(addRandomNoise))
        
    if balanceDataset or balanceAugmentations:
        print("  | .. Balancing dataset")
        if balanceDataset:
            print("  | .. .. the instances will be balanced so that an equal number of peak and background instances are present before augmentation")
        if balanceAugmentations:
            print("  | .. .. the instances will be balanced during instance augmentation. peaks or backgrounds underrepresented will be used several times more than the other class")
    
    print("  | .. Augmenting")
    if addRandomNoise:
        print("  | .. .. adding random noise")
        print("  | .. .. maxRandFactor: '%s'"%(maxRandFactor))
        print("  | .. .. maximum noise add level (relative to most abundant signal) '%s'"%(maxNoiseLevelAdd))
    if shiftRTs:
        print("  | .. .. shifting RTs of background instances")
        print("  | .. .. maxShift: '%s'"%(maxShift))
    print("  | .. Using each instance %d times for training"%(useEachInstanceNTimes))
    print("\n")
    
    
    print("PeakBotMRM configuration")
    print(PeakBotMRM.Config.getAsStringFancy())
    print("\n")
    
    ## Generate training instances
    validationDSs = []
    trainDataset = PeakBotMRM.MemoryDataset()
    for trainDS in trainDSs:
            
        print("Adding training dataset '%s'"%(trainDS["DSName"]))
        
        substances               = PeakBotMRM.loadTargets(trainDS["transitions"], 
                                                          excludeSubstances = trainDS["excludeSubstances"] if "excludeSubstances" in trainDS.keys() else None, 
                                                          includeSubstances = trainDS["includeSubstances"] if "includeSubstances" in trainDS.keys() else None, 
                                                          logPrefix = "  | ..")
        substances, integrations = PeakBotMRM.loadIntegrations(substances, 
                                                               trainDS["GTPeaks"], 
                                                               logPrefix = "  | ..")
        substances, integrations, sampleInfo = PeakBotMRM.loadChromatograms(substances, integrations, trainDS["samplesPath"],
                                                                            sampleUseFunction = trainDS["sampleUseFunction"] if "sampleUseFunction" in trainDS.keys() else None, 
                                                                            allowedMZOffset = allowedMZOffset, 
                                                                            MRMHeader = MRMHeader, 
                                                                            logPrefix = "  | ..")
        if showPeakMetrics:
            investigatePeakMetrics(expDir, substances, integrations, expName = "%s"%(trainDS["DSName"]), logPrefix = "  | ..")
        
        integrations = constrainAndBalanceDataset(balanceDataset, trainDS["checkPeakAttributes"] if "checkPeakAttributes" in trainDS.keys() else None, substances, integrations, logPrefix = "  | ..")
        
        dataset = exportOriginalInstancesForValidation(substances, integrations, "Train_Ori_%s"%(trainDS["DSName"]), logPrefix = "  | ..")
        dataset.shuffle()
        dataset.setName("%s_Train_Ori"%(trainDS["DSName"]))
        validationDSs.append(dataset)
        
        if useDSForTraining.lower() == "original":
            trainDataset.addData(dataset.data)
        
        if useDSForTraining.lower() == "augmented":
            dataset = generateAndExportAugmentedInstancesForTraining(substances, integrations, "Train_Aug_%s"%(trainDS["DSName"]), addRandomNoise, maxRandFactor, maxNoiseLevelAdd, 
                                                                     shiftRTs, maxShift, useEachInstanceNTimes, balanceAugmentations, 
                                                                     aug_augment, aug_OnlyUseAugmentationFromSamples, aug_addAugPeaks, aug_maxAugPeaksN, aug_plotAugInstances, 
                                                                     logPrefix = "  | ..")
            dataset.shuffle()
            dataset.setName("%s_Train_Aug"%(trainDS["DSName"]))
            
            if stratifyDataset:
                if intThres is not None:
                    use = dataset.data["inte.peak"][:,0] == 1
                    areas = dataset.data["inte.area"][use]
                    
                    if verbose: 
                        print(logPrefix, "  | .. There are %d peaks in the dataset, of which %d have smaller peak areas than %.0e (%.1f%%), while %d have peak areas exceeding %.0e (%.1f%%)"%(sum(use), sum(areas < intThres), intThres, sum(areas<intThres) / sum(use) * 100, sum(areas>=intThres), intThres, sum(areas>=intThres) / sum(use) * 100), sep="")
                        print(logPrefix, "  | .. .. This imbalance will be corrected by stratification of the overrepresented group", sep="")
                    
                    rat = sum(areas < intThres) / sum(areas >= intThres)
                    ovUse = [True for t in dataset.data["inte.peak"]]
                    for i in range(len(ovUse)):
                        if dataset.data["inte.peak"][i,0] == 1 and dataset.data["inte.area"][i] > intThres:
                            if random.random() >= rat:
                                ovUse[i] = False
                    dataset.useOrNotUse(ovUse)

                    use = dataset.data["inte.peak"][:,0] == 1
                    areas = dataset.data["inte.area"][use]
                    if verbose: 
                        print(logPrefix, "  | .. After stratification there are %d peaks in the dataset, of which %d have smaller peak areas than %.0e (%.1f%%), while %d have peak areas exceeding %.0e (%.1f%%)"%(sum(use), sum(areas < intThres), intThres, sum(areas<intThres) / sum(use) * 100, sum(areas>=intThres), intThres, sum(areas>=intThres) / sum(use) * 100), sep="")
                    
                if peakWidthThres is not None:
                    use = dataset.data["inte.peak"][:,0] == 1
                    widths = dataset.data["inte.rtEnd"][use] - dataset.data["inte.rtStart"][use]
                    
                    if verbose: 
                        print(logPrefix, "  | .. There are %d peaks in the dataset, of which %d have wider peaks than %.2f (%.1f%%), while %d have peaks narrower than %.2f (%.1f%%)"%(sum(use), sum(widths > peakWidthThres), peakWidthThres, sum(widths > peakWidthThres) / sum(use) * 100, sum(widths <= peakWidthThres), peakWidthThres, sum(widths <= peakWidthThres) / sum(use) * 100), sep="")
                        print(logPrefix, "  | .. .. This imbalance will be corrected by stratification of the overrepresented group", sep="")
                    
                    rat = sum(widths > peakWidthThres) / sum(widths <= peakWidthThres)
                    ovUse = [True for t in dataset.data["inte.peak"]]
                    for i in range(len(ovUse)):
                        if dataset.data["inte.peak"][i,0] == 1 and (dataset.data["inte.rtEnd"][i] - dataset.data["inte.rtStart"][i]) < peakWidthThres:
                            if random.random() >= rat:
                                ovUse[i] = False
                    dataset.useOrNotUse(ovUse)

                    use = dataset.data["inte.peak"][:,0] == 1
                    widths = dataset.data["inte.rtEnd"][use] - dataset.data["inte.rtStart"][use]
                    
                    if verbose: 
                        print(logPrefix, "  | .. There are %d peaks in the dataset, of which %d have wider peaks than %.2f (%.1f%%), while %d have peaks narrower than %.2f (%.1f%%)"%(sum(use), sum(widths > peakWidthThres), peakWidthThres, sum(widths > peakWidthThres) / sum(use) * 100, sum(widths <= peakWidthThres), peakWidthThres, sum(widths <= peakWidthThres) / sum(use) * 100), sep="")
                                
            trainDataset.addData(dataset.data)
        
        print("")

    ## prepare/split training dataset into train and validation set
    print("Dataset for training")
    tic()
    if useDSForTraining.lower() == "original":
        print("  | .. The original, unmodified instances are used")
    elif useDSForTraining.lower() == "augmented":
        print("  | .. The augmented instances are used")
    print("  | .. Shuffling training instances")
    trainDataset.shuffle()  
    print("  | .. took %.1f seconds"%(toc()))
    tic()  
    splitRatio = 0.7
    trainDataset, valDataset = trainDataset.split(ratio = splitRatio)
    trainDataset.setName("%s_split_train"%(expName))
    valDataset.setName("%s_split_val"%(expName))
    validationDSs.append(trainDataset)
    validationDSs.append(valDataset)
    print("  | .. Randomly split dataset into a training and validation dataset with %.1f and %.1f parts of the instances "%(splitRatio, 1-splitRatio))
    print("  | .. There are %d training (%s) and %d validation (%s) batches available"%(trainDataset.data["channel.rt"].shape[0], trainDataset.name, valDataset.data["channel.rt"].shape[0], valDataset.name))
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")
    
    
    ## add other datasets for validation metrics
    if valDSs is not None:
        for valDS in valDSs:
            
            print("Adding additional validation dataset '%s'"%(valDS["DSName"]))
            
            substances, integrations = PeakBotMRM.validate.getValidationSet(valDS, MRMHeader, allowedMZOffset)
            
            if len(integrations) > 0:
                
                if showPeakMetrics:
                    investigatePeakMetrics(expDir, substances, integrations, expName = "%s"%(valDS["DSName"]), logPrefix = "  | ..")
                
                dataset = exportOriginalInstancesForValidation(substances, integrations, "AddVal_Ori_%s"%(valDS["DSName"]), logPrefix = "  | ..")
                dataset.shuffle()
                dataset.setName("%s_AddVal_Ori"%(valDS["DSName"]))
                if dataset.getElements() > 0:
                    validationDSs.append(dataset)
                else:
                    print("\033[101m    !!! Not using validation dataset %s !!! \033[0m"%(valDS["DSName"]))
                    time.sleep(5)
        
            print("")
    
    print("Preparation for training took %.1f seconds"%(toc("Overall process")))
    print("")       
    
    ## Train new peakbotMRM model
    pb, chist, modelName = PeakBotMRM.trainPeakBotMRMModel(modelName = os.path.basename(modelFile), 
                                                           trainDataset = trainDataset,
                                                           addValidationDatasets = sorted(validationDSs, key=lambda x: x.name),
                                                           logBaseDir = logDir,
                                                           everyNthEpoch = -1, 
                                                           verbose = True)

    pb.saveModelToFile(modelFile)
    print("Newly trained PeakBotMRM saved to file '%s'"%(modelFile))
    print("\n")

    ## add current history
    chist["comment"] = comment
    chist["modelName"] = modelName
    chist["CreatedOn"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    chist["CreatedBy"] = platform.node()

    ### Summarize the training and validation metrices and losses
    with portalocker.Lock(historyFile, mode = "ab+", timeout = 60, check_interval = 2) as fh:
        try:
            fh.seek(0,0)
            history = pd.read_pickle(fh)
            history = pd.concat((history, chist), axis=0, ignore_index=True)
        except:
            print("Generating new history")
            history = chist
        fh.seek(0,0)
        fh.truncate(0)
        history.to_pickle(fh)

    print("\n\n\n")

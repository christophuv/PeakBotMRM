import logging
import tqdm
import os
import datetime
import platform
import numpy as np
import random
random.seed(2021)
import os
import natsort
import json

import PeakBotMRM
import PeakBotMRM.train
from PeakBotMRM.core import tic, toc, extractStandardizedEIC



def getPredictionSet(predDS, MRMHeader, allowedMZOffset):
    substances               = PeakBotMRM.loadTargets(predDS["transitions"], 
                                                      excludeSubstances = predDS["excludeSubstances"] if "excludeSubstances" in predDS else None, 
                                                      includeSubstances = predDS["includeSubstances"] if "includeSubstances" in predDS else None, 
                                                      logPrefix = "  | ..")
            
    substances, integrations, sampleInfo = PeakBotMRM.loadChromatograms(substances, None, 
                                                                        predDS["samplesPath"],
                                                                        sampleUseFunction = predDS["sampleUseFunction"] if "sampleUseFunction" in predDS else None, 
                                                                        allowedMZOffset = allowedMZOffset,
                                                                        MRMHeader = MRMHeader,
                                                                        logPrefix = "  | ..")
    
    return substances, integrations



def predictExperiment(expName, predDSs, modelFile, 
                      expDir = None, logDir = None, 
                      MRMHeader = None,
                      allowedMZOffset = 0.05,
                      oneRowHeader4Results = False):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
        
    if MRMHeader is None:
        MRMHeader = PeakBotMRM.Config.MRMHEADER
        

    
    logging.info("Predicting experiments")
    logging.info("  | .. Parameters")
    logging.info("  | .. .. expName: '%s'"%(expName))
    logging.info("  | .. .. modelFile: '%s'"%(modelFile))
    logging.info("  | .. .. expDir: '%s'"%(expDir))
    logging.info("  | .. .. logDir: '%s'"%(logDir))
    logging.info("  | .. .. MRMHeader: '%s'"%(MRMHeader))
    logging.info("  | .. .. allowedMZOffset: '%s'"%(allowedMZOffset))
    logging.info("\n")
    
    logging.info("PeakBotMRM configuration")
    logging.info(PeakBotMRM.Config.getAsStringFancy())
    logging.info("\n")


    ## administrative
    tic("Overall process")
    try:
        os.mkdir(expDir)
    except:
        pass
    try:
        os.mkdir(os.path.join(expDir, "SubstanceFigures"))
    except:
        pass
        
    for predDS in natsort.natsorted(predDSs, key = lambda x: x["DSName"]):
        
        logging.info("Predicting chromatographic peaks in dataset '%s'"%(predDS["DSName"]))
        
        substances, integrations = getPredictionSet(predDS, MRMHeader, allowedMZOffset)
        
        if len(integrations) > 0:

            predictDataset(modelFile, substances, integrations)
            acceptPredictions(integrations)
            
            substancesComments, samplesComments = calibrateIntegrations(substances, integrations)

            exportIntegrations(os.path.join(expDir, "%s_Results.tsv"%(predDS["DSName"])), 
                               substances, integrations, 
                               substancesComments = substancesComments, 
                               samplesComments = samplesComments, 
                               additionalCommentsForFile=[
                "PeakBot model: '%s'"%(modelFile)
            ], oneRowHeader4Results = oneRowHeader4Results)
            
            logging.info("\n\n")
    
    logging.info("All calculations took %.1f seconds"%(toc("Overall process")))

def predictDataset(modelFile, substances, integrations, callBackFunction = None, showConsoleProgress = True):
    logging.info("  | .. using model from '%s'"%(modelFile))
    pbModelPred = PeakBotMRM.loadModel(modelFile, mode="predict" , verbose = False)
    allSubstances = set()
    allSamples = set()
    
    calSamplesAndLevels = getCalibrationSamplesAndLevels(substances)
            
    subsN = set()
    sampN = set()
    for sub in integrations:
        subsN.add(sub)
        for samp in integrations[sub]:
            sampN.add(samp)
    logging.info("  | .. Processing %d samples with %d compounds"%(len(sampN), len(subsN)))
            
    used = {}
    for samp in sampN:
            calLevel = None
            if calSamplesAndLevels is not None:
                for samplePart, level in calSamplesAndLevels.items():
                    if samplePart in samp:
                        calLevel = level
            if calLevel is not None:
                if calLevel not in used:
                    used[calLevel] = []
                used[calLevel].append(samp)
    
    # TODO implement check here
    # raiseExp = False
    #for k, v in used.items():
    #    if len(v) > 1:
    #        logging.error("\33[91m  | .. Error: Calibration level '%s' found with multiple files. These are: \33[0m '%s'"%(k, "', '".join(v)))
    #        raiseExp = True
    #if raiseExp:
    #    logging.error("\33[91m  | .. Error: One or several calibration levels are not unique. Aborting...\33[0m")
    #    raise RuntimeError("Aborting to non-unique calibration levels")
    #elif len(used) < len(calSamplesAndLevels):
    #    logging.info("\33[93m  | .. Found %d of the %d provided calibration levels. Please double-check if these have been specified correctly\33[0m"%(len(used), len(calSamplesAndLevels)))
    #else:
    #    logging.info("  | .. Found all %d calibration levels"%(len(calSamplesAndLevels)))
            
    for substanceI, sub in tqdm.tqdm(enumerate(natsort.natsorted(integrations)), total = len(integrations), desc="  | .. predicting", disable = not showConsoleProgress):
        if callBackFunction is not None:
            callBackFunction(substanceI)
        if sub in integrations and len(integrations[sub]) > 0:
            allSubstances.add(sub)

            temp = {"channel.int"  : np.zeros((len(integrations[sub]), PeakBotMRM.Config.RTSLICES), dtype=float),
                    "channel.rts"  : np.zeros((len(integrations[sub]), PeakBotMRM.Config.RTSLICES), dtype=float)}
                    
            ## generate dataset for all samples of the current substance
            for samplei, samp in enumerate(integrations[sub]):
                allSamples.add(samp)
                inte = integrations[sub][samp]
                if inte is not None and inte.chromatogram is not None:
                    allSamples.add(samp)
                            
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]
                    refRT = substances[sub].refRT
                            
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                            
                    temp["channel.int"][samplei, :] = eicS
                    temp["channel.rts"][samplei, :] = rtsS
                                    
            ## predict and calculate metrics
            pred_peakTypes, pred_rtStartInds, pred_rtEndInds = PeakBotMRM.runPeakBotMRM(temp, model = pbModelPred, verbose = False)
                    
            ## inspect and summarize the results of the prediction and the metrics, optionally plot
            startRTs = []
            endRTs = []
            areas = []
            for samplei, samp in enumerate(integrations[sub]):
                inte = integrations[sub][samp] 
                        
                if inte is not None and inte.chromatogram is not None:
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]
                    refRT = substances[sub].refRT
                            
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                        
                     ## test if eic has detected signals
                    pred_isPeak     = 1 if pred_peakTypes[samplei] == 0 else 0
                    pred_rtStartInd = round(pred_rtStartInds[samplei])
                    pred_rtEndInd   = round(pred_rtEndInds[samplei])
                    
                    if pred_isPeak and PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT:
                        while pred_rtStartInd - 1 >= 0 and eicS[pred_rtStartInd - 1] <= eicS[pred_rtStartInd]:
                            pred_rtStartInd = pred_rtStartInd - 1
                        while pred_rtEndInd + 1 < eicS.shape[0] and eicS[pred_rtEndInd + 1] <= eicS[pred_rtEndInd]:
                            pred_rtEndInd = pred_rtEndInd + 1
                            
                        while pred_rtEndInd - 1 >= 0 and eicS[pred_rtEndInd - 1] <= eicS[pred_rtEndInd]:
                            pred_rtEndInd = pred_rtEndInd - 1
                        while pred_rtStartInd + 1 < eicS.shape[0] and eicS[pred_rtStartInd + 1] <= eicS[pred_rtStartInd]:
                            pred_rtStartInd = pred_rtStartInd + 1
                    
                    pred_rtStart    = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtStartInd))]
                    pred_rtEnd      = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtEndInd))]

                    inte.other["processed"]      = ""
                    inte.other["pred.type"]      = "PeakBotMRM"
                    inte.other["pred.comment"]   = "model '%s'"%(os.path.basename(modelFile))
                    if pred_isPeak:
                        inte.other["pred.foundPeak"] = 1
                        inte.other["pred.rtstart"]   = pred_rtStart
                        inte.other["pred.rtend"]     = pred_rtEnd
                        inte.other["pred.areaPB"]    = PeakBotMRM.integrateArea(eic, rts, inte.other["pred.rtstart"], inte.other["pred.rtend"])
                        
                        startRTs.append(pred_rtStart)
                        endRTs.append(pred_rtEnd)
                        areas.append(inte.other["pred.areaPB"])
                        
                    else:
                        inte.other["pred.foundPeak"] = 0
                        inte.other["pred.rtstart"]   = -1
                        inte.other["pred.rtend"]     = -1
                        inte.other["pred.areaPB"]    = -1
            
            if PeakBotMRM.Config.INTEGRATENOISE and len(startRTs) > 0:
                startRTs = np.array(startRTs)
                endRTs = np.array(endRTs)
                areas = np.array(areas)
                
                startNoiseInt = PeakBotMRM.core.weighted_percentile(startRTs, 1/np.log10(areas), PeakBotMRM.Config.INTEGRATENOISE_StartQuantile)
                endNoiseInt = PeakBotMRM.core.weighted_percentile(endRTs, 1/np.log10(areas), PeakBotMRM.Config.INTEGRATENOISE_EndQuantile)
                
                ## implement noise integration here
                ## todo improve this aspect by calculating baselines
                for samplei, samp in enumerate(integrations[sub]):
                    inte = integrations[sub][samp] 
                            
                    if inte is not None and inte.chromatogram is not None and inte.other["pred.foundPeak"] == 0:
                        rts = inte.chromatogram["rts"]
                        eic = inte.chromatogram["eic"]
                        refRT = substances[sub].refRT
                                
                        ## standardize EIC
                        rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                        
                        pred_rtStartInd = PeakBotMRM.core.arg_find_nearest(rtsS, startNoiseInt)
                        pred_rtEndInd   = PeakBotMRM.core.arg_find_nearest(rtsS, endNoiseInt)
                        
                        if PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT:
                            while pred_rtStartInd - 1 >= 0 and eicS[pred_rtStartInd - 1] <= eicS[pred_rtStartInd]:
                                pred_rtStartInd = pred_rtStartInd - 1
                            while pred_rtEndInd + 1 < eicS.shape[0] and eicS[pred_rtEndInd + 1] <= eicS[pred_rtEndInd]:
                                pred_rtEndInd = pred_rtEndInd + 1
                            
                            while pred_rtEndInd - 1 >= 0 and eicS[pred_rtEndInd - 1] <= eicS[pred_rtEndInd]:
                                pred_rtEndInd = pred_rtEndInd - 1
                            while pred_rtStartInd + 1 < eicS.shape[0] and eicS[pred_rtStartInd + 1] <= eicS[pred_rtStartInd]:
                                pred_rtStartInd = pred_rtStartInd + 1
                        
                        inte.other["pred.foundPeak"] = 2
                        inte.other["pred.rtstart"]   = rtsS[pred_rtStartInd]
                        inte.other["pred.rtend"]     = rtsS[pred_rtEndInd]
                        inte.other["pred.areaPB"]    = PeakBotMRM.integrateArea(eic, rts, inte.other["pred.rtstart"], inte.other["pred.rtend"])
                        
                
                        
                                                
def acceptPredictions(integrations):
    for subName in integrations:
        for sampName in integrations[subName]:
            inte = integrations[subName][sampName]
            inte.type = inte.other["pred.type"]
            inte.comment = inte.other["pred.comment"]
            inte.foundPeak = inte.other["pred.foundPeak"]
            inte.rtStart = inte.other["pred.rtstart"]
            inte.rtEnd = inte.other["pred.rtEnd"]
            inte.area = inte.other["pred.areaPB"]
            

def getCalibrationSamplesAndLevels(substances):
    calSamplesAndLevels = {}
    errors = 0
    for substanceName in substances: 
        if substances[substanceName].calSamples is not None: 
            for sampPart, level in substances[substanceName].calSamples.items():
                if level > 0:
                    if sampPart not in calSamplesAndLevels:
                        calSamplesAndLevels[sampPart] = level
                    else:
                        if calSamplesAndLevels[sampPart] != level:
                            logging.info("\33[91m  | .. Error: Calibration sample '%s' found with multiple levels '%s'\33[0m"%(sampPart, str(calSamplesAndLevels)))
                            logging.info(substances[substanceName].calSamples)
                            errors += 1
    if errors > 0:
        ## TODO improve errror message for user
        logging.error("\33[91m  | .. Error: Calibration levels are not unique among samples. Please double-check.\33[0m")
        raise RuntimeError("Error: Calibration levels are not unique among samples")
    return calSamplesAndLevels

def exportIntegrations(toFile, substances, integrations, substanceOrder = None, samplesOrder = None, substancesComments = None, samplesComments = None, sampleMetaData = None, oneRowHeader4Results = False, additionalCommentsForFile = None, separator = "\t"):
    for substanceName in integrations:
        isd = [substances[s].name for s in substances if substances[s].internalStandard == substanceName]
        if len(isd) > 0:
            if substancesComments is None or substanceName not in substancesComments or "Comment" not in substancesComments[substanceName]:
                substancesComments[substanceName]["Comment"] = ""
            else:
                substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "; "
            substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "ISTD for: '%s'"%(isd)
    
    calSamplesAndLevels = getCalibrationSamplesAndLevels(substances)
            
    ## generate results table
    logging.info("  | .. Generating results table (%s)"%(toFile))
    if substanceOrder is None:
        substanceOrder = natsort.natsorted(substances)
    if samplesOrder is None:
        allSamps = set()
        for s in substances:
            if s in integrations:
                for samp in integrations[s]:
                    allSamps.add(samp)
        samplesOrder = natsort.natsorted(list(allSamps))
    
    with open(toFile, "w") as fout:
        headersSample = ["", "", "Name", "Data File", "Type", "Level", "Acq. Date-Time", "Method", "Inj. volume", "Dilution", "Comment"]
        headersPerSubstance = ["Comment", "IntegrationType", "RT", "Int. Start", "Int. End", "Area", "ISTDRatio", "Final Conc.", "Conc. unit", "Accuracy"]
                
        if oneRowHeader4Results:
            fout.write(separator)  ## SampleName and CalibrationLevel
            for substanceName in substanceOrder:
                substanceName = substanceName.replace(separator, "--SEPARATOR--")
                for h in headersPerSubstance:
                    fout.write("%s%s"%(separator, h))
            fout.write("\n")
            
        else:
            ## Header row 1
            fout.write("Sample" + (separator*(len(headersSample)-1)))
            for substanceName in substanceOrder:
                fout.write(separator + (substanceName.replace("\"", "'").replace(separator, "--SEPARATOR--")) + (separator*(len(headersPerSubstance)-1)))
            fout.write("\n")

            ## Header row 2
            fout.write(separator.join(headersSample))
            for substanceName in substanceOrder:
                fout.write(separator + (separator.join(headersPerSubstance)))
            fout.write("\n")
                
        fout.write("#" + separator.join("" for k in headersSample))
        for substanceName in substanceOrder:
            for h in headersPerSubstance:
                fout.write(separator)
                if substanceName in substancesComments and h in substancesComments[substanceName]:
                    fout.write(("%s"%(json.dumps(substancesComments[substanceName][h]))).replace("\"", "'").replace(separator, "--SEPARATOR--"))
        fout.write("\n")
        pbComments = set()
        for sample in samplesOrder:
            sampleInfo = {}
            for k in headersSample:
                if sampleMetaData is not None and sample in sampleMetaData and k in sampleMetaData[sample]:
                    sampleInfo[k] = sampleMetaData[sample][k]
            sampleInfo[""] = "!"
            sampleInfo["Level"] = ""
            if calSamplesAndLevels is not None:
                for samplePart, level in calSamplesAndLevels.items():
                    if samplePart in sample:
                        sampleInfo["Level"] = str(level)
            fout.write(separator.join([sampleInfo[k].replace("\"", "'").replace(separator, "--SEPARATOR--") if k in sampleInfo else "" for k in headersSample]))
            
            for substanceName in substanceOrder:
                peakInfo = {}
                if substanceName in integrations and sample in integrations[substanceName] and integrations[substanceName][sample].chromatogram is not None:
                    temp = integrations[substanceName][sample]
                    if samplesComments is not None and substanceName in samplesComments and sample in samplesComments[substanceName]:
                        peakInfo["Comment"] = "%s"%("; ".join((str(s) for s in samplesComments[substanceName][sample])))
                    
                    peakInfo["Type"] = "%s"%(temp.type)
                    pbComments.add(temp.comment)
                    ## Prediction
                    if temp.foundPeak:
                        peakInfo["IntegrationType"]       = ["", "Peak", "Noise", "Manual"][temp.foundPeak]
                        peakInfo["Int. Start"]            = "%.3f"%(temp.rtStart)
                        peakInfo["Int. End"]              = "%.3f"%(temp.rtEnd)
                        peakInfo["Area"]                  = "%.3f"%(temp.area)
                        if temp.istdRatio is not None:
                            peakInfo["ISTDRatio"]         = "%f"  %(temp.istdRatio)
                        if temp.concentration is not None:
                            peakInfo["Final Conc."]       = "%.5f"%(temp.concentration)
                        peakInfo["Conc. unit"]            = substances[substanceName].calLevel1ConcentrationUnit
                else:
                    peakInfo["Comment"] = "not processed"
                fout.write(separator)
                fout.write(separator.join([peakInfo[k].replace("\"", "'").replace(separator, "--SEPARATOR--") if k in peakInfo else "" for k in headersPerSubstance]))

            fout.write("\n")
                
        ## include processing information in TSV file
        fout.write("## Status"); fout.write("\n")
        fout.write("## .. '' no problems"); fout.write("\n")
        fout.write("## .. 1 substance not processed as no respective channel was present in the sample"); fout.write("\n")
        fout.write("## .. 100 other problem"); fout.write("\n")
        fout.write("## Parameters"); fout.write("\n")
        fout.write("## PeakBotMRM configuration"); fout.write("\n")
        fout.write("## .. '" + ("'\n## .. '".join(PeakBotMRM.Config.getAsString().split(";"))) + "'"); fout.write("\n")
        fout.write("## General information"); fout.write("\n")
        fout.write("## .. Date: '%s'"%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))); fout.write("\n")
        fout.write("## .. Computer: '%s'"%(platform.node())); fout.write("\n")
        if additionalCommentsForFile is not None:
            fout.write("## Additional comments"); fout.write("\n")
            if additionalCommentsForFile is not None:
                for x in additionalCommentsForFile:
                    fout.write("## .. %s"%(x))
                    fout.write("\n")
            for x in pbComments:
                fout.write("## .. %s"%(x))
                fout.write("\n")

def calibrateIntegrations(substances, integrations):
    calSamplesAndLevels = getCalibrationSamplesAndLevels(substances)
    
    substancesComments = {}
    samplesComments = {}
    
    for substanceName in integrations:
        if substanceName not in substancesComments:
            substancesComments[substanceName] = {}
        
        if substanceName not in samplesComments:
            samplesComments[substanceName] = {}
            
            
        for samplei, sample in enumerate(integrations[substanceName]):
            inteSub = integrations[substanceName][sample]
            inteSub.concentration = None
            inteSub.istdRatio = None
            
        if substances[substanceName].internalStandard is not None:
            if "Comment" not in substancesComments[substanceName]:
                substancesComments[substanceName]["Comment"] = ""
            else:
                substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "; "
            substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "ISTD: '%s'"%(substances[substanceName].internalStandard)
                
        if substances[substanceName].calculateCalibration:
            calExp = []
            calObs = []
            calCal = []
            fromType = None
                    
            for samplei, sample in enumerate(integrations[substanceName]):
                if sample not in samplesComments[substanceName]:
                    samplesComments[substanceName][sample] = []
                    
                for samplePart, level in substances[substanceName].calSamples.items():
                    if samplePart in sample:
                        inteSub = integrations[substanceName][sample]
                        exp = None
                        obs = None
                        if substances[substanceName].internalStandard is not None and substances[substanceName].internalStandard in integrations:
                            inteIST = integrations[substances[substanceName].internalStandard][sample]
                            if inteSub is not None and inteSub.chromatogram is not None and inteIST is not None and inteIST.chromatogram is not None:
                                if inteSub.foundPeak in [1,3] and not np.isnan(inteSub.area) and inteIST.foundPeak in [1,3] and not np.isnan(inteIST.area):
                                    ratio = inteSub.area / inteIST.area
                                    inteSub.istdRatio = ratio
                                            
                                    exp = level * substances[substanceName].calLevel1Concentration
                                    obs = ratio
                                    fromType = "ISTD ratio"
                        else:
                            if inteSub is not None and inteSub.chromatogram is not None and inteSub.foundPeak in [1,3] and not np.isnan(inteSub.area):
                                exp = level * substances[substanceName].calLevel1Concentration
                                obs = inteSub.area
                                fromType = "Peak areas"
                                
                        if exp is not None and obs is not None and not np.isnan(exp) and not np.isnan(obs) and not np.isinf(exp) and not np.isinf(obs) and exp > 0 and obs > 0 and level > 0:
                            samplesComments[substanceName][sample].append("Cal: Using for cal. (exp. conc. %s)"%str(exp))
                            calExp.append(exp)
                            calObs.append(obs)
                        
            if len(calExp) > 1 and substances[substanceName].calculateCalibration:
                calcCal = False
                try:
                    model, r2, yhat, params, strRepr = PeakBotMRM.calibrationRegression(calObs, calExp, type = substances[substanceName].calibrationMethod)
                    calCal = model(np.array((calObs)).reshape(-1,1))
                    
                    calToOrigin = None
                    for i in range(len(calObs)):
                        if calToOrigin is None and calCal[i] > 0:
                            calToOrigin = i

                    substancesComments[substanceName]["Final Conc."] = {"R2": r2, "points": len(calObs), "fromType": fromType, "formula": strRepr, "method": substances[substanceName].calibrationMethod, "ConcentrationAtLevel1": str(substances[substanceName].calLevel1Concentration)}                    
                    calcCal = True
                    
                except Exception as ex:
                    logging.exception("Error during calibration calculation of substance '%s': call is PeakBotMRM.calibrationRegression(%s, %s, type = '%s')"%(substanceName, str(calObs), str(calExp), substances[substanceName].calibrationMethod))
                    raise ex
                
                if calcCal:                        
                    for samplei, sample in enumerate(integrations[substanceName]):
                        inteSub = integrations[substanceName][sample]
                        if substances[substanceName].internalStandard is not None and substances[substanceName].internalStandard in integrations:
                            if sample in integrations[substances[substanceName].internalStandard]:
                                inteIST = integrations[substances[substanceName].internalStandard][sample]
                                if inteSub is not None and inteSub.chromatogram is not None and inteIST is not None and inteIST.chromatogram is not None:
                                    if inteSub.foundPeak and not np.isnan(inteSub.area) and inteIST.foundPeak and not np.isnan(inteIST.area):
                                        ratio = inteSub.area / inteIST.area
                                        ratio = np.nan_to_num(ratio)
                                        inteSub.istdRatio = ratio
                                        
                                        calcConc = model(np.array((ratio)).reshape(-1,1))
                                        inteSub.concentration = calcConc
                                        
                                        if calToOrigin is not None and ratio < calObs[calToOrigin]:
                                            ## Idea of Ulrich Goldmann: linear interpolation to origin for the lowest calibration level
                                            ## Attention: Use with caution, not fully tested
                                            samplesComments[substanceName][sample].append("Outside: Ratio is lower than calibration range with a non-negative regression (level %.3f, observed %.3f). Linear interpolation from last non-negative calibration will be used. Use with caution"%(calExp[calToOrigin], calObs[calToOrigin]))
                                            inteSub.concentration = model(np.array((calObs[calToOrigin])).reshape(-1,1)) * ratio / calObs[calToOrigin]
                                            
                                        elif ratio > np.max(calObs):
                                            samplesComments[substanceName][sample].append("Outside: Ratio is higher than calibration range.")
                                    
                        elif inteSub is not None and inteSub.chromatogram is not None and inteSub.foundPeak and not np.isnan(inteSub.area):
                            calcConc = model(np.array((inteSub.area)).reshape(-1,1))
                            inteSub.concentration = calcConc
                                        
                            if calToOrigin is not None and inteSub.area < calObs[calToOrigin]:
                                ## Idea of Ulrich Goldmann: linear interpolation to origin for the lowest calibration level
                                ## Attention: Use with caution, not fully tested
                                samplesComments[substanceName][sample].append("Outside: Area is lower than calibration range with a non-negative regression (level %.3f, observed %.3f). Linear interpolation from last non-negative calibration will be used. Use with caution"%(calExp[calToOrigin], calObs[calToOrigin]))
                                inteSub.concentration = model(np.array((calObs[calToOrigin])).reshape(-1,1)) * inteSub.area / calObs[calToOrigin]
                                
                            elif inteSub.area > np.max(calObs):
                                samplesComments[substanceName][sample].append("Outside: Area is higher than calibration range.")
    
    return substancesComments, samplesComments

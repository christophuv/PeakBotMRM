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
                                                      excludeSubstances = predDS["excludeSubstances"] if "excludeSubstances" in predDS.keys() else None, 
                                                      includeSubstances = predDS["includeSubstances"] if "includeSubstances" in predDS.keys() else None, 
                                                      logPrefix = "  | ..")
            
    substances, integrations, sampleInfo = PeakBotMRM.loadChromatograms(substances, None, 
                                                                        predDS["samplesPath"],
                                                                        sampleUseFunction = predDS["sampleUseFunction"] if "sampleUseFunction" in predDS.keys() else None, 
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
    for substanceName in integrations.keys():
        subsN.add(substanceName)
        for sample in integrations[substanceName]:
            sampN.add(sample)
    logging.info("  | .. Processing %d samples with %d compounds"%(len(sampN), len(subsN)))
            
    used = {}
    for sample in sampN:
            calLevel = None
            if calSamplesAndLevels is not None:
                for samplePart, level in calSamplesAndLevels.items():
                    if samplePart in sample:
                        calLevel = level
            if calLevel is not None:
                if calLevel not in used.keys():
                    used[calLevel] = []
                used[calLevel].append(sample)
    
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
            
    for substanceI, substanceName in tqdm.tqdm(enumerate(natsort.natsorted(integrations.keys())), total = len(integrations.keys()), desc="  | .. predicting", disable = not showConsoleProgress):
        if callBackFunction is not None:
            callBackFunction(substanceI)
        if substanceName in integrations.keys() and len(integrations[substanceName]) > 0:
            allSubstances.add(substanceName)

            temp = {"channel.int"  : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.RTSLICES), dtype=float),
                            "channel.rts"  : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.RTSLICES), dtype=float)}
                    
            ## generate dataset for all samples of the current substance
            for samplei, sample in enumerate(integrations[substanceName].keys()):
                allSamples.add(sample)
                inte = integrations[substanceName][sample]
                if inte is not None and inte.chromatogram is not None:
                    allSamples.add(sample)
                            
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]
                    refRT = substances[substanceName].refRT
                            
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                            
                    temp["channel.int"][samplei, :] = eicS
                    temp["channel.rts"][samplei, :] = rtsS
                                    
            ## predict and calculate metrics
            pred_peakTypes, pred_rtStartInds, pred_rtEndInds = PeakBotMRM.runPeakBotMRM(temp, model = pbModelPred, verbose = False)
                    
            ## inspect and summarize the results of the prediction and the metrics, optionally plot
            for samplei, sample in enumerate(integrations[substanceName].keys()):
                inte = integrations[substanceName][sample] 
                        
                if inte is not None and inte.chromatogram is not None:
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]
                    refRT = substances[substanceName].refRT
                            
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                        
                     ## test if eic has detected signals
                    pred_isPeak     = pred_peakTypes[samplei] == 0
                    pred_rtStartInd = round(pred_rtStartInds[samplei])
                    pred_rtEndInd   = round(pred_rtEndInds[samplei])
                    
                    if True and pred_isPeak and PeakBotMRM.Config.EXTENDBORDERSUNTILINCREMENT:
                        while pred_rtStartInd - 1 >= 0 and eicS[pred_rtStartInd - 1] <= eicS[pred_rtStartInd]:
                            pred_rtStartInd = pred_rtStartInd - 1
                        while pred_rtEndInd + 1 < eicS.shape[0] and eicS[pred_rtEndInd + 1] <= eicS[pred_rtEndInd]:
                            pred_rtEndInd = pred_rtEndInd + 1
                    
                    pred_rtStart    = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtStartInd))]
                    pred_rtEnd      = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtEndInd))]

                    inte.other["processed"]      = ""
                    inte.other["pred.type"]      = "PeakBotMRM"
                    inte.other["pred.comment"]   = "model '%s'"%(os.path.basename(modelFile))
                    inte.other["pred.rtstart"]   = pred_rtStart
                    inte.other["pred.rtend"]     = pred_rtEnd
                    inte.other["pred.foundPeak"] = pred_isPeak
                    if inte.other["pred.foundPeak"]:
                        inte.other["pred.areaPB"] = PeakBotMRM.integrateArea(eic, rts, pred_rtStart, pred_rtEnd)
                    else:
                        inte.other["pred.areaPB"] = -1
                        
def acceptPredictions(integrations):
    for subName in integrations.keys():
        for sampName in integrations[subName].keys():
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
    for substanceName in integrations.keys():
        isd = [substances[s].name for s in substances if substances[s].internalStandard == substanceName]
        if len(isd) > 0:
            if substancesComments is None or substanceName not in substancesComments.keys() or "Comment" not in substancesComments[substanceName]:
                substancesComments[substanceName]["Comment"] = ""
            else:
                substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "; "
            substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "ISTD for: '%s'"%(isd)
    
    calSamplesAndLevels = getCalibrationSamplesAndLevels(substances)
            
    ## generate results table
    logging.info("  | .. Generating results table (%s)"%(toFile))
    if substanceOrder is None:
        substanceOrder = natsort.natsorted(substances.keys())
    if samplesOrder is None:
        allSamps = set()
        for s in substances.keys():
            if s in integrations.keys():
                for samp in integrations[s].keys():
                    allSamps.add(samp)
        samplesOrder = natsort.natsorted(list(allSamps))
    
    with open(toFile, "w") as fout:
        headersSample = ["", "", "Name", "Data File", "Type", "Level", "Acq. Date-Time", "Method", "Inj. volume", "Dilution", "Comment"]
        headersPerSubstance = ["Comment", "Type", "RT", "Int. Start", "Int. End", "Area", "ISTDRatio", "Final Conc.", "Accuracy"]
                
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
                if substanceName in substancesComments.keys() and h in substancesComments[substanceName].keys():
                    fout.write(("%s"%(json.dumps(substancesComments[substanceName][h]))).replace("\"", "'").replace(separator, "--SEPARATOR--"))
        fout.write("\n")
        pbComments = set()
        for sample in samplesOrder:
            sampleInfo = {}
            for k in headersSample:
                if sampleMetaData is not None and sample in sampleMetaData.keys() and k in sampleMetaData[sample].keys():
                    sampleInfo[k] = sampleMetaData[sample][k]
            sampleInfo[""] = "!"
            sampleInfo["Level"] = ""
            if calSamplesAndLevels is not None:
                for samplePart, level in calSamplesAndLevels.items():
                    if samplePart in sample:
                        sampleInfo["Level"] = str(level)
            fout.write(separator.join([sampleInfo[k].replace("\"", "'").replace(separator, "--SEPARATOR--") if k in sampleInfo.keys() else "" for k in headersSample]))
            
            for substanceName in substanceOrder:
                peakInfo = {}
                if substanceName in integrations.keys() and sample in integrations[substanceName].keys() and integrations[substanceName][sample].chromatogram is not None:
                    temp = integrations[substanceName][sample]
                    if samplesComments is not None and substanceName in samplesComments.keys() and sample in samplesComments[substanceName].keys():
                        peakInfo["Comment"] = "%s"%("; ".join((str(s) for s in samplesComments[substanceName][sample])))
                    
                    peakInfo["Type"] = "%s"%(temp.type)
                    pbComments.add(temp.comment)
                    ## Prediction
                    if temp.foundPeak:
                        peakInfo["Int. Start"]            = "%.3f"%(temp.rtStart)
                        peakInfo["Int. End"]              = "%.3f"%(temp.rtEnd)
                        peakInfo["Area"]                  = "%.3f"%(temp.area)
                        if temp.istdRatio is not None:
                            peakInfo["ISTDRatio"]         = "%f"  %(temp.istdRatio)
                        if temp.concentration is not None:
                            peakInfo["Final Conc."]       = "%.5f"%(temp.concentration)
                else:
                    peakInfo["Comment"] = "not processed"
                fout.write(separator)
                fout.write(separator.join([peakInfo[k].replace("\"", "'").replace(separator, "--SEPARATOR--") if k in peakInfo.keys() else "" for k in headersPerSubstance]))

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
    
    for substanceName in integrations.keys():
        if substanceName not in substancesComments.keys():
            substancesComments[substanceName] = {}
        
        if substances[substanceName].internalStandard is not None:
            if "Comment" not in substancesComments[substanceName]:
                substancesComments[substanceName]["Comment"] = ""
            else:
                substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "; "
            substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "ISTD: '%s'"%(substances[substanceName].internalStandard)
                
        if substances[substanceName].calculateCalibration:
            calExp = []
            calObs = []
            fromType = None
                    
            for samplei, sample in enumerate(integrations[substanceName].keys()):
                for samplePart, level in substances[substanceName].calSamples.items():
                    if samplePart in sample:
                        inteSub = integrations[substanceName][sample]
                        exp = None
                        obs = None
                        if substances[substanceName].internalStandard is not None and substances[substanceName].internalStandard in integrations.keys():
                            inteIST = integrations[substances[substanceName].internalStandard][sample]
                            if inteSub is not None and inteSub.chromatogram is not None and inteIST is not None and inteIST.chromatogram is not None:
                                if inteSub.foundPeak and not np.isnan(inteSub.area) and inteIST.foundPeak and not np.isnan(inteIST.area):
                                    ratio = inteSub.area / inteIST.area
                                    inteSub.istdRatio = ratio
                                            
                                    exp = level * substances[substanceName].calLevel1Concentration
                                    obs = ratio
                                    fromType = "ISTD ratio"
                        else:
                            if inteSub is not None and inteSub.chromatogram is not None and inteSub.foundPeak and not np.isnan(inteSub.area):
                                exp = level * substances[substanceName].calLevel1Concentration
                                obs = inteSub.area
                                fromType = "Peak areas"
                                
                        if exp is not None and obs is not None and not np.isnan(exp) and not np.isnan(obs) and exp > 0 and obs > 0 and level > 0:
                            if substanceName not in samplesComments.keys():
                                samplesComments[substanceName] = {}
                            if sample not in samplesComments[substanceName].keys():
                                samplesComments[substanceName][sample] = []
                            samplesComments[substanceName][sample].append("Using for cal. (exp. conc. %s)"%str(exp))
                            calExp.append(exp)
                            calObs.append(obs)
                        
            if len(calExp) > 1 and substances[substanceName].calculateCalibration:                    
                model, r2, yhat, params, strRepr = PeakBotMRM.calibrationRegression(calObs, calExp, type = substances[substanceName].calibrationMethod)
                substancesComments[substanceName]["Final Conc."] = {"R2": r2, "points": len(calObs), "fromType": fromType, "formula": strRepr, "method": substances[substanceName].calibrationMethod, "ConcentrationAtLevel1": str(substances[substanceName].calLevel1Concentration)}
                        
                for samplei, sample in enumerate(integrations[substanceName].keys()):
                    inteSub = integrations[substanceName][sample]
                    if substances[substanceName].internalStandard is not None and substances[substanceName].internalStandard in integrations.keys():
                        if sample in integrations[substances[substanceName].internalStandard].keys():
                            inteIST = integrations[substances[substanceName].internalStandard][sample]
                            if inteSub is not None and inteSub.chromatogram is not None and inteIST is not None and inteIST.chromatogram is not None:
                                if inteSub.foundPeak and not np.isnan(inteSub.area) and inteIST.foundPeak and not np.isnan(inteIST.area):
                                    ratio = inteSub.area / inteIST.area
                                    ratio = np.nan_to_num(ratio)
                                    inteSub.istdRatio = ratio
                                    
                                    calcConc = model(np.array((ratio)).reshape(-1,1))
                                    inteSub.concentration = calcConc
                    else:
                        if inteSub is not None and inteSub.chromatogram is not None and inteSub.foundPeak and not np.isnan(inteSub.area):
                            calcConc = model(np.array((inteSub.area)).reshape(-1,1))
                            inteSub.concentration = calcConc
    
    return substancesComments, samplesComments

## General imports
from email.errors import HeaderParseError
from sklearn.metrics import calinski_harabasz_score
import tqdm
import os
import datetime
import platform
import numpy as np
from sklearn.linear_model import LinearRegression
import random
random.seed(2021)
import os
import natsort
import json

import PeakBotMRM
import PeakBotMRM.train
from PeakBotMRM.core import tic, toc, extractStandardizedEIC
print("\n")



def getPredictionSet(predDS, MRMHeader, allowedMZOffset):
    substances               = PeakBotMRM.loadTargets(predDS["transitions"], 
                                                      excludeSubstances = predDS["excludeSubstances"] if "excludeSubstances" in predDS.keys() else None, 
                                                      includeSubstances = predDS["includeSubstances"] if "includeSubstances" in predDS.keys() else None, 
                                                      logPrefix = "  | ..")
        
    substances, integrations = PeakBotMRM.loadChromatograms(substances, None, 
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
                      oneRowHeader4Results = False, 
                      calSamplesAndLevels = None, 
                      calSubstanceConcentrations = None,
                      internalStandards = None, 
                      calLevelsForCompounds = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
        
    if MRMHeader is None:
        MRMHeader = PeakBotMRM.Config.MRMHEADER
        
    if calSamplesAndLevels is None:
        calSamplesAndLevels = {}
    if calSubstanceConcentrations is None:
        calSubstanceConcentrations = {}
    if internalStandards is None:
        internalStandards = {}
    if calLevelsForCompounds is None:
        calLevelsForCompounds = {}
        
    
    print("Predicting experiments")
    print("  | .. Parameters")
    print("  | .. .. expName: '%s'"%(expName))
    print("  | .. .. modelFile: '%s'"%(modelFile))
    print("  | .. .. expDir: '%s'"%(expDir))
    print("  | .. .. logDir: '%s'"%(logDir))
    print("  | .. .. MRMHeader: '%s'"%(MRMHeader))
    print("  | .. .. allowedMZOffset: '%s'"%(allowedMZOffset))
    print("\n")
    
    print("PeakBotMRM configuration")
    print(PeakBotMRM.Config.getAsStringFancy())
    print("\n")


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
        
        print("Predicting chromatographic peaks in dataset '%s'"%(predDS["DSName"]))
        
        substances, integrations = getPredictionSet(predDS, MRMHeader, allowedMZOffset)
        if len(integrations) > 0:

            print("  | .. using model from '%s'"%(modelFile))
            pbModelPred = PeakBotMRM.loadModel(modelFile, mode="predict" , verbose = False)
            allSubstances = set()
            allSamples = set()
            
            subsN = set()
            sampN = set()
            for substanceName in integrations.keys():
                subsN.add(substanceName)
                for sample in integrations[substanceName]:
                    sampN.add(sample)
            print("  | .. Processing %d samples with %d compounds"%(len(sampN), len(subsN)))
            
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
            raiseExp = False
            for k, v in used.items():
                if len(v) > 1:
                    print("\33[91m  | .. Error: Calibration level '%s' found with multiple files. These are: \33[0m '%s'"%(k, "', '".join(v)))
                    raiseExp = True
            if raiseExp:
                print("\33[91m  | .. Error: One or several calibration levels are not unique. Aborting...\33[0m")
                raise RuntimeError("Aborting to non-unique calibration levels")
            elif len(used) < len(calSamplesAndLevels):
                print("\33[93m  | .. Found %d of the %d provided calibration levels. Please double-check if these have been specified correctly\33[0m"%(len(used), len(calSamplesAndLevels)))
            else:
                print("  | .. Found all %d calibration levels"%(len(calSamplesAndLevels)))
            
            substancesComments = {}            
            for substanceName in tqdm.tqdm(natsort.natsorted(integrations.keys()), desc="  | .. predicting"):
                substancesComments[substanceName] = {}
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
                            pred_rtStart    = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtStartInd))]
                            pred_rtEnd      = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtEndInd))]

                            inte.other["processed"]      = ""
                            inte.other["pred.rtstart"]   = pred_rtStart
                            inte.other["pred.rtend"]     = pred_rtEnd
                            inte.other["pred.foundPeak"] = pred_isPeak
                            if inte.other["pred.foundPeak"]:
                                inte.other["pred.areaPB"] = PeakBotMRM.integrateArea(eic, rts, pred_rtStart, pred_rtEnd)
                            else:
                                inte.other["pred.areaPB"] = -1
            
            for substanceName in integrations.keys():
                isd = None
                if isd is not None and substanceName in internalStandards.keys():
                    print("\33[93m  | .. Error: Substance '%s' has several different internal standards configured. Please check this and correct the problem. \33[0m"%(substanceName))
                    raise RuntimeError("Error: Substance '%s' has several different internal standards configured"%(substanceName))
                if substanceName in internalStandards.keys():
                    isd = internalStandards[substanceName]
                    if "Comment" not in substancesComments[substanceName]:
                        substancesComments[substanceName]["Comment"] = ""
                    else:
                        substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "; "
                    substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "ISTD: '%s'"%(isd)
                
                calExp = []
                calObs = []
                
                for samplei, sample in enumerate(integrations[substanceName].keys()):
                    for samplePart, level in calSamplesAndLevels.items():
                        if samplePart in sample and (substanceName not in calLevelsForCompounds.keys() or calLevelsForCompounds[substanceName] == "all" or calLevelsForCompounds[substanceName][0] <= level <= calLevelsForCompounds[substanceName][1]):
                            inteSub = integrations[substanceName][sample]
                            exp = None
                            obs = None
                            if isd is not None:
                                inteIST = integrations[isd][sample]
                                if inteSub is not None and inteSub.chromatogram is not None and inteIST is not None and inteIST.chromatogram is not None:
                                    if inteSub.other["pred.foundPeak"] and not np.isnan(inteSub.other["pred.areaPB"]) and inteIST.other["pred.foundPeak"] and not np.isnan(inteIST.other["pred.areaPB"]):
                                        ratio = inteSub.other["pred.areaPB"] / inteIST.other["pred.areaPB"]
                                        inteSub.other["pred.ISTDRatio"] = ratio
                                        
                                        exp = level * calSubstanceConcentrations[substanceName] if substanceName in calSubstanceConcentrations.keys() else level
                                        obs = ratio
                            else:
                                if inteSub is not None and inteSub.chromatogram is not None and inteSub.other["pred.foundPeak"] and not np.isnan(inteSub.other["pred.areaPB"]):
                                    exp = level * calSubstanceConcentrations[substanceName] if substanceName in calSubstanceConcentrations.keys() else level
                                    obs = inteSub.other["pred.areaPB"]
                            
                            if exp is not None and obs is not None and not np.isnan(exp) and not np.isnan(obs):
                                calExp.append(exp)
                                calObs.append(obs)
                    
                if len(calExp) > 1:
                    
                    model, r2, intercept, coef, calObshat = PeakBotMRM.calibrationRegression(calObs, calExp)
                    substancesComments[substanceName]["RelativeConcentration"] = {"R2": r2, "points": len(calObs), "intercept": intercept, "coef": coef, "method": PeakBotMRM.Config.CALIBRATIONMETHOD}
                    
                    for samplei, sample in enumerate(integrations[substanceName].keys()):
                        inteSub = integrations[substanceName][sample]
                        if isd is not None:
                            inteIST = integrations[isd][sample]
                            if inteSub is not None and inteSub.chromatogram is not None and inteIST is not None and inteIST.chromatogram is not None:
                                if inteSub.other["pred.foundPeak"] and not np.isnan(inteSub.other["pred.areaPB"]) and inteIST.other["pred.foundPeak"] and not np.isnan(inteIST.other["pred.areaPB"]):
                                    ratio = inteSub.other["pred.areaPB"] / inteIST.other["pred.areaPB"]
                                    inteSub.other["pred.ISTDRatio"] = ratio
                                    
                                    calPre = model.predict(np.array((ratio)).reshape(-1,1))
                                    inteSub.other["pred.level"] = calPre
                        else:
                            if inteSub is not None and inteSub.chromatogram is not None and inteSub.other["pred.foundPeak"] and not np.isnan(inteSub.other["pred.areaPB"]):
                                calPre = model.predict(np.array((inteSub.other["pred.areaPB"])).reshape(-1,1))
                                inteSub.other["pred.level"] = calPre

            for substanceName in integrations.keys():
                if substanceName in internalStandards.values():
                    isd = list(internalStandards.keys())[list(internalStandards.values()).index(substanceName)]
                    if "Comment" not in substancesComments[substanceName]:
                        substancesComments[substanceName]["Comment"] = ""
                    else:
                        substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "; "
                    substancesComments[substanceName]["Comment"] = substancesComments[substanceName]["Comment"] + "ISTD for: '%s'"%(isd)
            
            
            ## generate results table
            print("  | .. Generating results table (%s)"%(os.path.join(expDir, "%s_Results.tsv"%(predDS["DSName"]))))
            substanceOrder = natsort.natsorted(list(allSubstances))
            samplesOrder = natsort.natsorted(list(allSamples))
            with open(os.path.join(expDir, "%s_Results.tsv"%(predDS["DSName"])), "w") as fout:
                headersPerSubstance = ["Comment", "PeakStart", "PeakEnd", "PeakAreaPB", "ISTDRatio", "RelativeConcentration"]
                
                if oneRowHeader4Results:
                    fout.write("\t")  ## SampleName and CalibrationLevel
                    for substanceName in substanceOrder:
                        substanceName = substanceName.replace("\t", "--TAB--")
                        for h in headersPerSubstance:
                            fout.write("\t%s"%(h))
                    fout.write("\n")
                else:
                    ## Header row 1
                    fout.write("\t")
                    for substanceName in substanceOrder:
                        fout.write("\t%s%s"%(substanceName.replace("\t", "--TAB--"), "\t"*(len(headersPerSubstance)-1)))
                    fout.write("\n")

                    ## Header row 2
                    fout.write("Sample\tRelativeConcentrationLevel")
                    for substanceName in substanceOrder:
                        fout.write("\t" + ("\t".join(headersPerSubstance)))
                    fout.write("\n")
                
                fout.write("#\t")
                for substanceName in substanceOrder:
                    for h in headersPerSubstance:
                        fout.write("\t")
                        if substanceName in substancesComments.keys() and h in substancesComments[substanceName].keys():
                            fout.write("%s"%(json.dumps(substancesComments[substanceName][h])))
                fout.write("\n")

                for sample in samplesOrder:
                    fout.write("%s"%(sample, ))
                    calLevel = ""
                    if calSamplesAndLevels is not None:
                        for samplePart, level in calSamplesAndLevels.items():
                            if samplePart in sample:
                                calLevel = str(level)
                    fout.write("\t")
                    fout.write(calLevel)
                    for substanceName in substanceOrder:
                        if substanceName in integrations.keys() and sample in integrations[substanceName].keys() and integrations[substanceName][sample].chromatogram is not None:
                            substanceInfo = {}
                            temp = integrations[substanceName][sample]
                            if temp.other["processed"] == '':
                                ## Prediction
                                if temp.other["pred.foundPeak"]:
                                    substanceInfo["PeakStart"] = "%.3f"%(temp.other["pred.rtstart"]) if not np.isnan(temp.other["pred.rtstart"]) else -1
                                    substanceInfo["PeakEnd"] = "%.3f"%(temp.other["pred.rtend"])   if not np.isnan(temp.other["pred.rtend"])   else -1
                                    substanceInfo["PeakAreaPB"] ="%.3f"%(temp.other["pred.areaPB"])  if not np.isnan(temp.other["pred.areaPB"])  else -1
                                    substanceInfo["RelativeConcentration"] = "%.5f"%(temp.other["pred.level"])   if "pred.level" in temp.other.keys() else "ASDf"
                                if "pred.ISTDRatio" in temp.other.keys():
                                    substanceInfo["ISTDRatio"] = "%f"%(temp.other["pred.ISTDRatio"])
                            else:
                                substanceInfo["Comment"] = temp.other["processed"]
                        else:
                            substanceInfo["Comment"] = temp.other["processed"]
                        fout.write("\t")
                        fout.write("\t".join([substanceInfo[k] if k in substanceInfo.keys() else "" for k in headersPerSubstance]))
                    fout.write("\n")
                
                ## include processing information in TSV file
                fout.write("## Status"); fout.write("\n")
                fout.write("## .. '' no problems"); fout.write("\n")
                fout.write("## .. 1 substance not processed as no respective channel was present in the sample"); fout.write("\n")
                fout.write("## .. 100 other problem"); fout.write("\n")
                fout.write("## Parameters"); fout.write("\n")
                fout.write("## .. expName: '%s'"%(expName)); fout.write("\n")
                fout.write("## .. modelFile: '%s'"%(modelFile)); fout.write("\n")
                fout.write("## .. expDir: '%s'"%(expDir)); fout.write("\n")
                fout.write("## .. logDir: '%s'"%(logDir)); fout.write("\n")
                fout.write("## .. MRMHeader: '%s'"%(MRMHeader)); fout.write("\n")
                fout.write("## .. allowedMZOffset: '%s'"%(allowedMZOffset)); fout.write("\n")
                fout.write("## PeakBotMRM configuration"); fout.write("\n")
                fout.write("## .. '" + ("'\n## .. '".join(PeakBotMRM.Config.getAsString().split(";"))) + "'"); fout.write("\n")
                fout.write("## General information"); fout.write("\n")
                fout.write("## .. Date: '%s'"%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))); fout.write("\n")
                fout.write("## .. Computer: '%s'"%(platform.node())); fout.write("\n")
            
            print("\n\n")
    
    print("All calculations took %.1f seconds"%(toc("Overall process")))

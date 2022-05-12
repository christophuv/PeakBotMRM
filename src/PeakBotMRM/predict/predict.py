## General imports
import tqdm
import os
import datetime
import platform
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(2021)
import os
import math


import PeakBotMRM
import PeakBotMRM.train
from PeakBotMRM.core import tic, toc, extractStandardizedEIC
print("\n")



def getPredictionSet(valDS, MRMHeader, allowedMZOffset):
    substances               = PeakBotMRM.loadTargets(valDS["transitions"], 
                                                      excludeSubstances = valDS["excludeSubstances"] if "excludeSubstances" in valDS.keys() else None, 
                                                      includeSubstances = valDS["includeSubstances"] if "includeSubstances" in valDS.keys() else None, 
                                                      logPrefix = "  | ..")
        
    substances, integrations = PeakBotMRM.loadChromatograms(substances, None, 
                                                            valDS["samplesPath"],
                                                            sampleUseFunction = valDS["sampleUseFunction"] if "sampleUseFunction" in valDS.keys() else None, 
                                                            allowedMZOffset = allowedMZOffset,
                                                            MRMHeader = MRMHeader,
                                                            logPrefix = "  | ..")
    return substances, integrations



def predictExperiment(expName, predDSs, modelFile, 
                      expDir = None, logDir = None, 
                      MRMHeader = None,
                      allowedMZOffset = 0.05,
                      oneRowHeader4Results = False, 
                      calSamplesAndLevels = {},
                      plotSubstance = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
        
    if MRMHeader is None:
        MRMHeader = PeakBotMRM.Config.MRMHEADER
        
    
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
        
    for predDS in predDSs:
        
        print("Evaluating validation dataset '%s'"%(predDS["DSName"]))
        
        substances, integrations = getPredictionSet(predDS, MRMHeader, allowedMZOffset)
                
        if len(integrations) > 0:

            print("  | .. using model from '%s'"%(modelFile))
            offsetEIC = 0.2
            offsetRT1 = 0
            offsetRT2 = 0
            offsetRTMod = 1
            pbModelPred = PeakBotMRM.loadModel(modelFile, mode="predict" , verbose = False)
            allSubstances = set()
            allSamples = set()
            
            subsN = set()
            sampN = set()
            for substanceName in integrations.keys():
                subsN.add(substanceName)
                for sample in integrations[substanceName]:
                    sampN.add(sample)
            
            for substanceName in tqdm.tqdm(integrations.keys(), desc="  | .. validating"):
                allSubstances.add(substanceName)
                if plotSubstance == "all" or substanceName in plotSubstance:
                    fig, ((ax1, ax2), (ax5, ax6), (ax9, ax10)) = plt.subplots(3,2, sharey = "row", sharex = True, gridspec_kw = {'height_ratios':[2,1,1]})
                    fig.set_size_inches(15, 16)

                temp = {"channel.int"  : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.RTSLICES), dtype=float),
                        "channel.rts"  : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.RTSLICES), dtype=float)}
                
                ## generate dataset for all samples of the current substance
                for samplei, sample in enumerate(integrations[substanceName].keys()):
                    inte = integrations[substanceName][sample]
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

                    inte.other["pred.rtstart"]   = pred_rtStart
                    inte.other["pred.rtend"]     = pred_rtEnd
                    inte.other["pred.foundPeak"] = pred_isPeak
                    if inte.other["pred.foundPeak"]:
                        inte.other["pred.areaPB"] = PeakBotMRM.integrateArea(eic, rts, pred_rtStart, pred_rtEnd)
                    else:
                        inte.other["pred.areaPB"] = -1
                    
                    if plotSubstance == "all" or substanceName in plotSubstance:
                        ## plot results; find correct axis to plot to 
                        ax = ax1 if pred_isPeak else ax2
                        axR = ax5 if pred_isPeak else ax6
                        axS = ax9 if pred_isPeak else ax10
                                            
                        ax.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), 
                                 max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], 
                                 [offsetEIC*samplei, 
                                 offsetEIC*samplei], 
                                 "slategrey", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        axR.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), 
                                 max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], 
                                 [0,
                                 0], 
                                 "slategrey", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        axS.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), 
                                 max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], 
                                 [0,
                                 0], 
                                 "slategrey", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        
                        ## plot raw, scaled data according to classification prediction and integration result
                        b = min(eic)
                        m = max([i-b for i in eic])
                        ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], 
                                [(e-b)/m+offsetEIC*samplei for e in eic], 
                                "lightgrey", linewidth=.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], 
                                        [(e-b)/m+offsetEIC*samplei for e in eic], 
                                        offsetEIC*samplei, 
                                        facecolor='w', lw=0, zorder=(len(integrations[substanceName].keys())-samplei+1)*2-1)
                        ## add detected peak
                        if pred_isPeak:
                            ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if pred_rtStart <= t <= pred_rtEnd], 
                                    [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if pred_rtStart <= rts[i] <= pred_rtEnd], 
                                    "olivedrab", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                            ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if pred_rtStart <= t <= pred_rtEnd], 
                                            [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if pred_rtStart <= rts[i] <= pred_rtEnd], 
                                            offsetEIC*samplei, 
                                            facecolor='yellowgreen', lw=0, zorder=(len(integrations[substanceName].keys())-samplei+1)*2-1)
                                                    
                        ## plot raw data
                        axR.plot(rts, eic, "lightgrey", linewidth=.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        ## add detected peak
                        if pred_isPeak:
                            axR.plot([t for t in rts if pred_rtStart <= t <= pred_rtEnd], 
                                    [e for i, e in enumerate(eic) if pred_rtStart <= rts[i] <= pred_rtEnd], 
                                    "olivedrab", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)

                        ## plot scaled data
                        ## add detected peak
                        minInt = 0
                        maxInt = 1
                        if np.sum(eicS) > 0:
                            minInt = min([e for e in eicS if e > 0])
                            maxInt = max([e-minInt for e in eicS])
                        axS.plot(rts, 
                                [(e-minInt)/maxInt for e in eic], 
                                "lightgrey", linewidth=.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        if pred_isPeak:
                            axS.plot([t for t in rts if pred_rtStart <= t <= pred_rtEnd], 
                                    [(e-minInt)/maxInt for i, e in enumerate(eic) if pred_rtStart <= rts[i] <= pred_rtEnd], 
                                    "olivedrab", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)

                if plotSubstance == "all" or substanceName in plotSubstance:
                    ## add retention time of peak
                    for ax in [ax1, ax2, ax5, ax6, ax9, ax10]:
                        ax.axvline(x = substances[substanceName].refRT, zorder = 1E6, alpha = 0.2)

                    ## add title and scale accordingly
                    for ax in [ax1, ax2]:
                        ax.set(xlabel = 'time (min)', ylabel = 'rel. abundance')
                        ax.set_ylim(-0.2, len(integrations[substanceName].keys()) * offsetEIC + 1 + 0.2)
                    for ax in [ax5, ax6]:
                        ax.set(xlabel = 'time (min)', ylabel = 'abundance')
                    for ax in [ax9, ax10]:
                        ax.set_ylim(-0.1, 1.1)
                    ax1.set_title('Prediciton peak', loc="left")
                    ax2.set_title('Prediction no peak', loc="left")
                    fig.suptitle('%s\n%s\n%s\nGreen EIC and area: prediction; grey EIC: standardized EIC; light grey EIC: raw data'%(substanceName), fontsize=14)

                    plt.tight_layout()
                    fig.savefig(os.path.join(expDir, "SubstanceFigures","%s_%s.png"%(predDS["DSName"], substanceName)), dpi = 600)
                    plt.close(fig)
            
            ## generate results table
            print("  | .. Generating results table (%s)"%(os.path.join(expDir, "%s_Results.tsv"%(predDS["DSName"]))))
            allSubstances = list(allSubstances)
            allSamples = list(allSamples)
            with open(os.path.join(expDir, "%s_Results.tsv"%(predDS["DSName"])), "w") as fout:
                if oneRowHeader4Results:
                    fout.write("\t")  ## SampleName and CalibrationLevel
                    for substanceName in allSubstances:
                        substanceName = substanceName.replace("\t", "--TAB--")
                        fout.write("\t%s.PeakStart\t%s.PeakEnd\t%s.PeakAreaPB\t%s.PeakRelativeConcentration"%(substanceName, substanceName, substanceName, substanceName))
                    fout.write("\n")
                else:
                    ## Header row 1
                    fout.write("\t")
                    for substanceName in allSubstances:
                        fout.write("\t%s\t%s\t%s"%(substanceName.replace("\t", "--TAB--"), "","",""))
                    fout.write("\n")

                    ## Header row 2
                    fout.write("Sample\tRelativeConcentrationLevel")
                    for substanceName in allSubstances:
                        fout.write("\tPeakStart\tPeakEnd\tPeakAreaPB\tPeakRelativeConcentration")
                    fout.write("\n")

                for sample in allSamples:
                    fout.write(sample)
                    fout.write("\t")
                    calLevel = ""
                    if calSamplesAndLevels is not None:
                        for samplePart, level in calSamplesAndLevels.items():
                            if samplePart in sample:
                                calLevel = str(level)
                    fout.write(calLevel)
                    for substanceName in allSubstances:
                        if substanceName in integrations.keys() and sample in integrations[substanceName].keys() and integrations[substanceName][sample].chromatogram is not None:
                            temp = integrations[substanceName][sample]
                            
                            ## Prediction
                            if temp.other["pred.foundPeak"]:
                                fout.write("\t%s\t%s\t%s"%("%.3f"%(temp.other["pred.rtstart"]) if not np.isnan(temp.other["pred.rtstart"]) else -1, 
                                                           "%.3f"%(temp.other["pred.rtend"])   if not np.isnan(temp.other["pred.rtend"])   else -1, 
                                                           "%.3f"%(temp.other["pred.areaPB"])  if not np.isnan(temp.other["pred.areaPB"])  else -1))
                            else:
                                fout.write("\t\t\t")
                        else:
                            fout.write("\t\t\t")
                    fout.write("\n")
                
                ## include processing information in TSV file
                fout.write("## Parameters"); fout.write("\n")
                fout.write("## .. expName: '%s'"%(expName)); fout.write("\n")
                fout.write("## .. modelFile: '%s'"%(modelFile)); fout.write("\n")
                fout.write("## .. expDir: '%s'"%(expDir)); fout.write("\n")
                fout.write("## .. logDir: '%s'"%(logDir)); fout.write("\n")
                fout.write("## .. MRMHeader: '%s'"%(MRMHeader)); fout.write("\n")
                fout.write("## .. allowedMZOffset: '%s'"%(allowedMZOffset)); fout.write("\n")
                fout.write("## PeakBotMRM configuration"); fout.write("\n")
                fout.write("## .. '" + ("'\n## .. '".join(PeakBotMRM.Config.getAsStringFancy().split("\n"))) + "'"); fout.write("\n")
                fout.write("## General information"); fout.write("\n")
                fout.write("## .. Date: '%s'"%(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))); fout.write("\n")
                fout.write("## .. Computer: '%s'"%(platform.node())); fout.write("\n")
            
            print("\n\n")
    
    print("All calculations took %.1f seconds"%(toc("Overall process")))

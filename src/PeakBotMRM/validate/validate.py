## General imports
from re import I
import tqdm
import os
import pickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotnine as p9
from scipy.cluster.hierarchy import median, leaves_list
from scipy.spatial.distance import pdist
import pandas as pd
import random
random.seed(2021)
import os
import math


import PeakBotMRM
import PeakBotMRM.train
from PeakBotMRM.core import tic, toc, extractStandardizedEIC, getInteRTIndsOnStandardizedEIC
print("\n")



def getValidationSet(valDS, MRMHeader, allowedMZOffset):
    substances               = PeakBotMRM.loadTargets(valDS["transitions"], 
                                                      excludeSubstances = valDS["excludeSubstances"] if "excludeSubstances" in valDS.keys() else None, 
                                                      includeSubstances = valDS["includeSubstances"] if "includeSubstances" in valDS.keys() else None, 
                                                      logPrefix = "  | ..")
    
    substances, integrations = PeakBotMRM.loadIntegrations(substances, 
                                                           valDS["GTPeaks"], 
                                                           logPrefix = "  | ..")
    
    substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, 
                                                            valDS["samplesPath"],
                                                            sampleUseFunction = valDS["sampleUseFunction"] if "sampleUseFunction" in valDS.keys() else None, 
                                                            allowedMZOffset = allowedMZOffset,
                                                            MRMHeader = MRMHeader,
                                                            logPrefix = "  | ..")
    
    integrations = PeakBotMRM.train.constrainAndBalanceDataset(False, 
                                                               valDS["checkPeakAttributes"] if "checkPeakAttributes" in valDS.keys() else None, 
                                                               substances, 
                                                               integrations, 
                                                               logPrefix = "  | ..")
        
    return substances,integrations



def validateExperiment(expName, valDSs, modelFile, 
                       expDir = None, logDir = None, 
                       MRMHeader = None,
                       allowedMZOffset = 0.05,
                       plotSubstance = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
        
    if MRMHeader is None:
        MRMHeader = PeakBotMRM.Config.MRMHEADER
        
    
    print("Validating experiments")
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
        
    metricsTable = {}

    for valDS in valDSs:
        
        print("Evaluating validation dataset '%s'"%(valDS["DSName"]))
        
        substances, integrations = getValidationSet(valDS, MRMHeader, allowedMZOffset)
                
        if len(integrations) > 0:
            
            PeakBotMRM.train.investigatePeakMetrics(expDir, substances, integrations, expName = "%s"%(valDS["DSName"]))

            print("  | .. using model from '%s'"%(modelFile))
            offsetEIC = 0.2
            offsetRT1 = 0
            offsetRT2 = 0
            offsetRTMod = 1
            pbModelPred = PeakBotMRM.loadModel(modelFile, mode="predict" , verbose = False)
            pbModelEval = PeakBotMRM.loadModel(modelFile, mode="training", verbose = False)
            allSubstances = set()
            allSamples = set()
            
            subsN = set()
            sampN = set()
            for substanceName in integrations.keys():
                subsN.add(substanceName)
                for sample in integrations[substanceName]:
                    sampN.add(sample)
            perInstanceResults = np.ones((len(subsN), len(sampN)), dtype=float)
            perInstanceResultsPD = []
            perInstanceResultsSubstances = list(subsN)
            perInstanceResultsSamples = list(sampN) 
            
            for substanceName in tqdm.tqdm(integrations.keys(), desc="  | .. validating"):
                allSubstances.add(substanceName)
                if plotSubstance == "all" or substanceName in plotSubstance:
                    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3,4, sharey = "row", sharex = True, gridspec_kw = {'height_ratios':[2,1,1]})
                    fig.set_size_inches(15, 16)

                temp = {"channel.int"  : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.RTSLICES), dtype=float),
                        "channel.rts"  : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.RTSLICES), dtype=float),
                        "inte.peak"    : np.zeros((len(integrations[substanceName]), PeakBotMRM.Config.NUMCLASSES), dtype=int),
                        "inte.rtInds"  : np.zeros((len(integrations[substanceName]), 2), dtype=float),
                        }
                truth = np.zeros((4))
                agreement = np.zeros((4))
                
                ## generate dataset for all samples of the current substance
                for samplei, sample in enumerate(integrations[substanceName].keys()):
                    inte = integrations[substanceName][sample]
                    allSamples.add(sample)
                    
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]
                    refRT = substances[substanceName].refRT
                    
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                    
                    ## get integration results on standardized area
                    bestRTInd, gt_isPeak, gt_rtStartInd, gt_rtEndInd, gt_rtStart, gt_rtEnd = \
                        getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    inte.foundPeak, 
                                                    inte.rtStart, 
                                                    inte.rtEnd)
                    
                    temp["channel.int"][samplei, :] = eicS
                    temp["channel.rts"][samplei, :] = rtsS
                    temp["inte.peak"][samplei,0]    = 1 if gt_isPeak else 0
                    temp["inte.peak"][samplei,1]    = 1 if not gt_isPeak else 0
                    temp["inte.rtInds"][samplei, 0] = gt_rtStartInd
                    temp["inte.rtInds"][samplei, 1] = gt_rtEndInd
                
                ## save some results to a file for testing
                if substanceName == "Adenosine monophosphate":
                    with open("out.pickle", "wb") as handle:
                        pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                ## predict and calculate metrics
                pred_peakTypes, pred_rtStartInds, pred_rtEndInds = PeakBotMRM.runPeakBotMRM(temp, model = pbModelPred, verbose = False)
                metrics                                          = PeakBotMRM.evaluatePeakBotMRM(temp, model = pbModelEval, verbose = False)
                
                ## inspect and summarize the results of the prediction and the metrics, optionally plot
                for samplei, sample in enumerate(integrations[substanceName].keys()):
                    inte = integrations[substanceName][sample] 
                    
                    rts = inte.chromatogram["rts"]
                    eic = inte.chromatogram["eic"]
                    refRT = substances[substanceName].refRT
                    
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                
                    ## get integration results on standardized area
                    bestRTInd, gt_isPeak, gt_rtStartInd, gt_rtEndInd, gt_rtStart, gt_rtEnd = \
                        getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    inte.foundPeak, 
                                                    inte.rtStart, 
                                                    inte.rtEnd)
                        
                    ## test if eic has detected signals
                    pred_isPeak     = pred_peakTypes[samplei] == 0
                    pred_rtStartInd = round(pred_rtStartInds[samplei])
                    pred_rtEndInd   = round(pred_rtEndInds[samplei])
                    pred_rtStart    = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtStartInd))]
                    pred_rtEnd      = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, pred_rtEndInd))]
                    truth[0] = truth[0] + (1 if gt_isPeak else 0)
                    truth[3] = truth[3] + (1 if not gt_isPeak else 0)
                    agreement[0] = agreement[0] + (1 if gt_isPeak and pred_isPeak else 0)
                    agreement[1] = agreement[1] + (1 if gt_isPeak and not pred_isPeak else 0)
                    agreement[2] = agreement[2] + (1 if not gt_isPeak and pred_isPeak else 0)
                    agreement[3] = agreement[3] + (1 if not gt_isPeak and not pred_isPeak else 0)

                    inte.other["pred.rtstart"]   = pred_rtStart
                    inte.other["pred.rtend"]     = pred_rtEnd
                    inte.other["pred.foundPeak"] = pred_isPeak
                    if inte.other["pred.foundPeak"]:
                        inte.other["pred.areaPB"] = PeakBotMRM.integrateArea(eic, rts, pred_rtStart, pred_rtEnd)
                    else:
                        inte.other["pred.areaPB"] = -1
                    if gt_isPeak:
                        inte.other["gt.areaPB"] = PeakBotMRM.integrateArea(eic, rts, gt_rtStart, gt_rtEnd)
                    else:
                        inte.other["gt.areaPB"] = -1
                    
                    ## generate heatmap matrix
                    val = 0
                    if gt_isPeak:
                        if pred_isPeak:
                            ## manual and prediction report a peak, x% correct
                            if inte.other["gt.areaPB"] < inte.other["pred.areaPB"]:
                                val = min(1, inte.other["pred.areaPB"] / inte.other["gt.areaPB"] - 1)
                            else:
                                val = -min(1, inte.other["gt.areaPB"] / inte.other["pred.areaPB"] - 1)
                        else:
                            ## manual integration reports a peak, but prediction does not, 100% incorrect
                            val = -1.00001
                    else:
                        if pred_isPeak:
                            ## manual integration reports not a peak, but prediction does, 100% incorrect
                            val = 1.00001
                        else:
                            ## both (manual and prediction) report not a peak, 100% correct
                            val = 0
                            
                    if val > 1.001 or val < -1.001:
                        print(substanceName, sample, gt_isPeak, inte["gt.areaPB"], gt_rtStart, gt_rtEnd, pred_isPeak, inte.other["pred.areaPB"], pred_rtStart, pred_rtEnd)
                    rowInd = perInstanceResultsSubstances.index(substanceName)
                    colInd = perInstanceResultsSamples.index(sample)
                    perInstanceResults[rowInd, colInd] = val
                    perInstanceResultsPD.append((substanceName, sample, val, 
                                                gt_isPeak , inte.rtStart               , inte.rtEnd              , inte.other["gt.areaPB"]     , inte.area,   
                                                pred_isPeak, inte.other["pred.rtstart"], inte.other["pred.rtend"], inte.other["pred.areaPB"]))
                    
                    if plotSubstance == "all" or substanceName in plotSubstance:
                        ## plot results; find correct axis to plot to 
                        ax = ax1
                        axR = ax5
                        axS = ax9
                        if gt_isPeak:
                            ax = ax1 if pred_isPeak else ax2
                            axR = ax5 if pred_isPeak else ax6
                            axS = ax9 if pred_isPeak else ax10
                        else:
                            ax = ax3 if pred_isPeak else ax4
                            axR = ax7 if pred_isPeak else ax8
                            axS = ax11 if pred_isPeak else ax12
                                            
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
                        ## add integration results
                        if gt_isPeak:            
                            ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if gt_rtStart <= t <= gt_rtEnd], 
                                    [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if gt_rtStart <= rts[i] <= gt_rtEnd], 
                                    "k", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                                                    
                        ## plot raw data
                        axR.plot(rts, eic, "lightgrey", linewidth=.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        ## add detected peak
                        if pred_isPeak:
                            axR.plot([t for t in rts if pred_rtStart <= t <= pred_rtEnd], 
                                    [e for i, e in enumerate(eic) if pred_rtStart <= rts[i] <= pred_rtEnd], 
                                    "olivedrab", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)
                        ## add integration results
                        if gt_isPeak:            
                            axR.plot([t for t in rts if gt_rtStart <= t <= gt_rtEnd], 
                                    [e for i, e in enumerate(eic) if gt_rtStart <= rts[i] <= gt_rtEnd], 
                                    "k", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)

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
                        if gt_isPeak:
                            axS.plot([t for t in rts if gt_rtStart <= t <= gt_rtEnd], 
                                    [(e-minInt)/maxInt for i, e in enumerate(eic) if gt_rtStart <= rts[i] <= gt_rtEnd], 
                                    "k", linewidth=0.25, zorder=(len(integrations[substanceName].keys())-samplei+1)*2)

            
                if substanceName not in metricsTable.keys():
                    metricsTable[substanceName] = {}
                metricsTable[substanceName] = {"CCA"          : metrics["pred.peak_categorical_accuracy"], 
                                            "MCC"          : metrics["pred.peak_MatthewsCorrelationCoefficient"],
                                            "RtMSE"        : metrics["pred.rtInds_MSE"], 
                                            "EICIOUPeaks"  : metrics["pred_EICIOUPeaks"],
                                            "Acc4Peaks"    : metrics["pred.peak_Acc4Peaks"],
                                            "Acc4NonPeaks" : metrics["pred.peak_Acc4NonPeaks"]}
                if substanceName in integrations.keys():
                    intes = []
                    intesD = []
                    preds = []
                    for sample in allSamples:
                        if sample in integrations[substanceName].keys():
                            intes.append(integrations[substanceName][sample].area)
                            intesD.append(integrations[substanceName][sample].other["gt.areaPB"])
                            preds.append(integrations[substanceName][sample].other["pred.areaPB"])
                    corr = np.corrcoef(intes, preds)[1,0]
                    metricsTable[substanceName]["CorrIP"] = corr
                    corr = np.corrcoef(intesD, preds)[1,0]
                    metricsTable[substanceName]["CorrIdP"] = corr
                
                if plotSubstance == "all" or substanceName in plotSubstance:
                    ## add retention time of peak
                    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
                        ax.axvline(x = substances[substanceName].refRT, zorder = 1E6, alpha = 0.2)

                    ## add title and scale accordingly
                    for ax in [ax1, ax2, ax3, ax4]:
                        ax.set(xlabel = 'time (min)', ylabel = 'rel. abundance')
                        ax.set_ylim(-0.2, len(integrations[substanceName].keys()) * offsetEIC + 1 + 0.2)
                    for ax in [ax5, ax6, ax7, ax8]:
                        ax.set(xlabel = 'time (min)', ylabel = 'abundance')
                    for ax in [ax9, ax10, ax11, ax12]:
                        ax.set_ylim(-0.1, 1.1)
                    total = agreement[0] + agreement[1] + agreement[2] + agreement[3]
                    ax1.set_title('Integration peak\nPrediciton peak\n%d (%.1f%%, integration %.1f%%)'%(agreement[0], agreement[0]/total*100, truth[0]/total*100), loc="left")
                    ax2.set_title('Integration peak\nPrediction no peak\n%d (%.1f%%)'%(agreement[1], agreement[1]/total*100), loc="left")
                    ax3.set_title('Integration no peak\nPrediction peak\n%d (%.1f%%)'%(agreement[2], agreement[2]/total*100), loc="left")
                    ax4.set_title('Integration no peak\nPrediction no peak\n%d (%.1f%%, integration %.1f%%)'%(agreement[3], agreement[3]/total*100, truth[3]/total*100), loc="left")
                    fig.suptitle('%s\n%s\n%s\nGreen EIC and area: prediction; black EIC: manual integration; grey EIC: standardized EIC; light grey EIC: raw data'%(substanceName, substances[substanceName].peakForm + "; " + substances[substanceName].rtShift + "; " + substances[substanceName].note, "; ".join("%s: %.4g"%(k, v) for k, v in metricsTable[substanceName].items())), fontsize=14)

                    plt.tight_layout()
                    fig.savefig(os.path.join(expDir, "SubstanceFigures","%s_%s.png"%(valDS["DSName"], substanceName)), dpi = 600)
                    plt.close(fig)
            
            
            with open(os.path.join(expDir, "%s_AllResults.pickle"%(valDS["DSName"])), "wb") as fout:
                pickle.dump((perInstanceResults, perInstanceResultsPD, perInstanceResultsSamples, perInstanceResultsSubstances), fout)
            
            indResPD = pd.DataFrame(perInstanceResultsPD, columns = ["substance", "sample", "value", 
                                                                    "GTPeak", "GTRTStart", "GTRTEnd", "GTAreaPB", "GTArea", 
                                                                    "PBPeak", "PBRTStart", "PBRTEnd", "PBArea"])
            
            ## generate results table
            if True:
                print("  | .. generating results table")
                allSubstances = list(allSubstances)
                allSamples = list(allSamples)
                with open(os.path.join(expDir, "%s_Results.tsv"%(valDS["DSName"])), "w") as fout:
                    fout.write("Sample")
                    for substanceName in allSubstances:
                        fout.write("\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(substanceName, "","","", "","",""))
                    fout.write("\n")

                    fout.write("")
                    for substanceName in allSubstances:
                        fout.write("\tGT.Start\tGT.End\tGT.Area\tGT.AreaPB\tPred.Start\tPred.End\tPred.AreaPB")
                    fout.write("\n")

                    for sample in allSamples:
                        fout.write(sample)
                        for substanceName in allSubstances:
                            if substanceName in integrations.keys() and sample in integrations[substanceName].keys() and integrations[substanceName][sample].chromatogram is not None:
                                temp = integrations[substanceName][sample]
                                ## Integration
                                if temp.area > 0:
                                    fout.write("\t%.3f\t%.3f\t%d\t%.3f"%(temp.rtStart, temp.rtEnd, temp.area, temp.other["gt.areaPB"]))
                                else:
                                    fout.write("\t\t\t\t")
                                ## Prediction
                                if temp.other["pred.foundPeak"]:
                                    fout.write("\t%.3f\t%.3f\t%.3f"%(temp.other["pred.rtstart"] if not np.isnan(temp.other["pred.rtstart"]) else -1, 
                                                                    temp.other["pred.rtend"]   if not np.isnan(temp.other["pred.rtend"])   else -1, 
                                                                    temp.other["pred.areaPB"]  if not np.isnan(temp.other["pred.areaPB"])  else -1))
                                else:
                                    fout.write("\t\t\t")
                            else:
                                fout.write("\t\t\t\t\t\t\t")
                        fout.write("\n")


            ## Plot metrics
            if False:
                print("  | .. plotting metrics")
                with open(os.path.join(expDir, "%s_metricsTable.pickle"%(valDS["DSName"])), "wb") as fout:
                    pickle.dump(metricsTable, fout)
                with open(os.path.join(expDir, "%s_metricsTable.pickle"%(valDS["DSName"])), "rb") as fin:
                    metricsTable = pickle.load(fin)

                rows = [i for i in metricsTable[list(metricsTable.keys())[0]].keys()]
                cols = [j for j in metricsTable.keys()]
                heatmapData = np.zeros((len(rows), len(cols)))
                for i, metric in enumerate(rows):
                    for j, c in enumerate(cols):
                        heatmapData[i,j] = metricsTable[c][metric]

                    fig, ax = plt.subplots()
                    fig.set_size_inches(32, 8)
                    
                    if metric=="PeakCA":
                        colors = ['firebrick', 'orange', 'dodgerblue', 'yellowgreen', 'olivedrab']
                        levels = [0,0.8, 0.9, 0.95, 0.99,1]
                        cmap= mpl.colors.ListedColormap(colors)
                        norm = mpl.colors.BoundaryNorm(levels, cmap.N)
                        im = ax.imshow(heatmapData[[i],:], cmap=cmap, norm=norm)
                    else:
                        im = ax.imshow(heatmapData[[i],:])
                    
                heatmapData = heatmapData[[i for i, k in enumerate(rows) if k != "RtMSE"],:]
                rows = [r for r in rows if r != "RtMSE"]
                
                fig, ax = plt.subplots()
                fig.set_size_inches(32, 8)
                
                if metric=="PeakCA":
                    colors = ['firebrick', 'orange', 'dodgerblue', 'yellowgreen', 'olivedrab']
                    levels = [0,0.8, 0.9, 0.95, 0.99,1]
                    cmap= mpl.colors.ListedColormap(colors)
                    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
                    im = ax.imshow(heatmapData, cmap=cmap, norm=norm)
                else:
                    im = ax.imshow(heatmapData)
                    
                plt.colorbar(im)
                ax.set_xticks(np.arange(len(cols)))
                ax.set_yticks(np.arange(len(rows)))
                ax.set_xticklabels(cols)
                ax.set_yticklabels(rows)
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
                plt.tight_layout()
                fig.savefig(os.path.join(expDir,"%s_Metrics.png"%(valDS["DSName"])), dpi = 300)
                plt.close()
                    
                    
                heatmapData = heatmapData[[i for i, r in enumerate(rows) if r in ["CCA", "EICIOUPeaks", "CorrIP", "CorrIdP"]],:]
                rows = [r for r in rows if r in ["CCA", "EICIOUPeaks", "CorrIP", "CorrIdP"]]
                
                fig, ax = plt.subplots()
                fig.set_size_inches(32, 8)
                
                if metric=="PeakCA":
                    colors = ['firebrick', 'orange', 'dodgerblue', 'yellowgreen', 'olivedrab']
                    levels = [0,0.8, 0.9, 0.95, 0.99,1]
                    cmap= mpl.colors.ListedColormap(colors)
                    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
                    im = ax.imshow(heatmapData, cmap=cmap, norm=norm)
                else:
                    im = ax.imshow(heatmapData)
                    
                plt.colorbar(im)
                ax.set_xticks(np.arange(len(cols)))
                ax.set_yticks(np.arange(len(rows)))
                ax.set_xticklabels(cols)
                ax.set_yticklabels(rows)
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
                plt.tight_layout()
                fig.savefig(os.path.join(expDir,"%s_Metrics_short.png"%(valDS["DSName"])), dpi = 300)
                plt.close()
            
            
            ## show correlation between areas integrated with GT method and PeakBot's method using the manual integration peak borders
            if False:
                print("  | .. plotting correlation of integration results 1/2")
                temp = indResPD.copy()
                temp = temp[temp["GTPeak"]]
                plot = (p9.ggplot(temp, p9.aes('np.log2(GTArea)', 'np.log2(GTAreaPB)', colour='substance'))
                    + p9.geom_point(alpha=0.1)
                    #+ p9.geom_smooth(method='lm')
                    + p9.theme(legend_position="none")
                    + p9.ggtitle("Comparison of GT and PeakBot integration methods based on the manual integration peak borders")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compManInte1.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
                
                ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
                plot = (p9.ggplot(temp, p9.aes('np.log2(GTArea)', 'np.log2(GTAreaPB / GTArea)', colour='substance'))
                    + p9.geom_point(alpha=0.1)
                    #+ p9.geom_smooth(method='lm')
                    + p9.theme(legend_position="none")
                    + p9.ggtitle("Comparison of GT and PeakBot integration methods based on the manual integration peak borders")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compManInte2.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
                
                ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
                temp2 = temp.copy()
                temp2 = temp2.drop(["substance"], axis=1)
                plot = (p9.ggplot(temp, p9.aes('np.log2(GTArea)', 'np.log2(GTAreaPB / GTArea)'))
                    + p9.geom_point(data = temp2, alpha=0.1)
                    + p9.geom_point(alpha=0.3, colour="firebrick")
                    #+ p9.geom_smooth(method='lm')
                    + p9.facets.facet_wrap("substance")
                    + p9.ggtitle("Comparison of GT and PeakBot integration methods based on the manual integration peak borders")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compManInte3.png"%(valDS["DSName"])), width=60, height=60, dpi=100, limitsize=False, verbose=False)
            
            
            ## show correlation between areas integrated with PeakBot's method using the manual integration peak borders and the predicted peak borders
            if False:
                print("  | .. plotting correlation of integration results 2/2")
                temp = indResPD.copy()
                temp = temp[temp["GTPeak"] & (temp["PBPeak"])]
                plot = (p9.ggplot(temp, p9.aes('np.log2(GTAreaPB)', 'np.log2(PBArea)', colour='substance'))
                    + p9.geom_point(alpha=0.1)
                    #+ p9.geom_smooth(method='lm')
                    + p9.ggtitle("Comparison of GT and PeakBot calculated areas using PB integration method")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compIntePred1.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
                
                ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
                plot = (p9.ggplot(temp, p9.aes('np.log2(GTAreaPB)', 'np.log2(GTAreaPB / PBArea)', colour='substance'))
                    + p9.geom_point(alpha=0.1)
                    + p9.theme(legend_position="none")
                    + p9.ggtitle("Comparison of GT and PeakBot calculated areas using PB integration method")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compIntePred2.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
                
                temp2 = temp.copy()
                temp2 = temp2.drop(["substance"], axis=1)
                
                ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
                plot = (p9.ggplot(temp, p9.aes('np.log2(GTAreaPB)', 'np.log2(GTAreaPB / PBArea)'))
                    + p9.geom_point(data = temp2, alpha=0.1)
                    + p9.geom_point(alpha=0.3, colour="firebrick")
                    + p9.facets.facet_wrap("substance")
                    + p9.ggtitle("Comparison of GT and PeakBot calculated areas using PB integration method")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compIntePred3.png"%(valDS["DSName"])), width=60, height=60, dpi=100, limitsize=False, verbose=False)
            
                
            ## Peak width comparison of manual integration and prediction
            if False:
                print("  | .. plotting peak width comparison")
                temp = indResPD.copy()
                temp = temp[(temp["GTPeak"]) & (temp["PBPeak"])]
                plot = (p9.ggplot(temp, p9.aes('GTRTEnd - GTRTStart', 'PBRTEnd - PBRTStart'))
                    + p9.geom_point(alpha=0.1)
                    + p9.geom_smooth(method='lm')
                    + p9.geom_abline(intercept = 0, slope=1)
                    + p9.ggtitle("Comparison of GT and PeakBot peak widths")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compPeakWidths.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
                
                plot = (p9.ggplot(temp, p9.aes('GTRTEnd - GTRTStart', 'PBRTEnd - PBRTStart'))
                    + p9.geom_point(alpha=0.1)
                    + p9.geom_smooth(method='lm')
                    + p9.geom_abline(intercept = 0, slope=1)
                    + p9.xlim(0,1.5) + p9.ylim(0,1.5)
                    + p9.ggtitle("Comparison of GT and PeakBot peak widths")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compPeakWidths_z.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
                
                temp2 = temp.copy()
                temp2 = temp2.drop(["substance"], axis=1)
                plot = (p9.ggplot(temp, p9.aes('GTRTEnd - GTRTStart', 'PBRTEnd - PBRTStart'))
                    + p9.geom_abline(intercept = 0, slope=1)
                    + p9.geom_point(data = temp2, alpha=0.1)
                    + p9.geom_point(alpha=0.3, colour="firebrick")
                    + p9.facets.facet_wrap("substance", scales="free")
                    + p9.theme(subplots_adjust={'hspace': 0.25, 'wspace': 0.25})
                    + p9.ggtitle("Comparison of GT and PeakBot peak widths")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compPeakWidths_f1.png"%(valDS["DSName"])), width=60, height=60, dpi=100, limitsize=False, verbose=False)
                
                plot = (p9.ggplot(temp, p9.aes(x='GTRTStart', y='GTRTEnd', xend='PBRTStart', yend='PBRTEnd', colour='substance'))
                    + p9.geom_abline(intercept = 0, slope=1, colour="firebrick")
                    + p9.geom_segment(alpha=0.3)
                    + p9.theme(legend_position="none")
                    + p9.xlab("Start (min)") + p9.ylab("End (min)")
                    + p9.ggtitle("Comparison of GT and PeakBot peak widths")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compPeakWidths2.png"%(valDS["DSName"])), width=30, height=30, dpi=300, limitsize=False, verbose=False)
                
                plot = (p9.ggplot(temp, p9.aes(x='GTRTStart', y='GTRTEnd', xend='PBRTStart', yend='PBRTEnd'))
                    + p9.geom_abline(intercept = 0, slope=1)
                    + p9.geom_segment(data = temp2, alpha=0.1)
                    + p9.geom_segment(alpha=0.3, colour="firebrick")
                    + p9.facets.facet_wrap("substance")
                    + p9.theme(legend_position="none")
                    + p9.xlab("Start (min)") + p9.ylab("End (min)")
                    + p9.ggtitle("Comparison of GT and PeakBot peak widths")
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_compPeakWidths2_f.png"%(valDS["DSName"])), width=60, height=60, dpi=100, limitsize=False, verbose=False)
                
            
            ## Hierarchical clustering and heatmap overview of results
            if True:
                print("  | .. generating comprehensive overview of validation experiment")
                ordRow = leaves_list(median(pdist(perInstanceResults, 'euclidean')))
                ordCol = leaves_list(median(pdist(np.transpose(perInstanceResults), 'euclidean')))
                
                indResPD = indResPD.assign(sampleHC = pd.Categorical(indResPD["sample"], categories = [perInstanceResultsSamples[i] for i in ordCol]))
                indResPD = indResPD.assign(substanceHC = pd.Categorical(indResPD["substance"], categories = [perInstanceResultsSubstances[i] for i in ordRow]))
                indResPD["value"] = round(indResPD["value"], 5)

                plot = (p9.ggplot(indResPD, p9.aes('substanceHC', 'sampleHC', fill='value'))
                    + p9.geom_tile(p9.aes(width=.95, height=.95))
                    + p9.geom_text(p9.aes(label='value'), size=1)
                    + p9.theme(axis_text_x = p9.element_text(rotation=45, hjust=1))
                    + p9.scales.scale_fill_gradientn(["Firebrick", "#440154FF", "#21908CFF", "Ghostwhite", "Ghostwhite", "Ghostwhite", "#21908CFF", "#FDE725FF", "Orange"], [(i+1.00001)/2.00002 for i in [-1.00001, -1, -0.101, -0.1, 0, 0.1, 0.101, 1, 1.00001]])
                    + p9.ggtitle(expName + ": Heatmap of predicted and manually derived integration results.\n0 (white) indicates the perfect agreement (manual and prediction for peak/nopeak agree and identical peak areas (+/- 10%) if peaks were detected)\n1.001 (red) indicates a predicted peak but nopeak in the manual integration, while -1.001 (orange) indicates a nopeak in the prediction but a manually integrated peak\ncolors between -1 and 1 indicate the increase (positive) or decrease (negative) of the abundance difference relative manually integrated peak area (in %)\n" + "PBPeak & GTPeak %d (%.1f%%); PBNoPeak & GTNoPeak %d (%.1f%%); PBPeak & GTNoPeak %d (%.1f%%); PBNoPeak & GTPeak %d (%.1f%%)\n"%(sum(indResPD["PBPeak"] & indResPD["GTPeak"]), sum(indResPD["PBPeak"] & indResPD["GTPeak"]) / indResPD.shape[0] * 100, sum(~indResPD["PBPeak"] & ~indResPD["GTPeak"]), sum(~indResPD["PBPeak"] & ~indResPD["GTPeak"])/ indResPD.shape[0] * 100, sum(indResPD["PBPeak"] & ~indResPD["GTPeak"]), sum(indResPD["PBPeak"] & ~indResPD["GTPeak"]) / indResPD.shape[0] * 100, sum(~indResPD["PBPeak"] & indResPD["GTPeak"]), sum(~indResPD["PBPeak"] & indResPD["GTPeak"]) / indResPD.shape[0] * 100) + str(round(indResPD[indResPD["PBPeak"] & indResPD["GTPeak"]]["value"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),2,)).replace("\n", "; ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " "))
                )
                p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_AllResults.pdf"%(valDS["DSName"])), width=50, height=20, limitsize=False, verbose=False)
            
            
            print("\n\n")
    
    print("All calculations took %.1f seconds"%(toc("Overall process")))

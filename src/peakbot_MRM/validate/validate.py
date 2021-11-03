




## General imports
import re
import tqdm
import os
import pickle
import pickle
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
random.seed(2021)
import os
import math

import pymzml

import peakbot_MRM
from peakbot_MRM.core import tic, toc, TabLog, readTSVFile, parseTSVMultiLineHeader
print("\n")


def extractStandardizedEIC(eic, rts, refRT):
    ## Find best rt-reference match and extract EIC as standardized EIC around that rt
    eicrts = np.array([abs(t - refRT) for t in rts])
    bestRTInd = np.argmin(eicrts)
    rtsS = np.zeros(peakbot_MRM.Config.RTSLICES, dtype=float)
    eicS = np.zeros(peakbot_MRM.Config.RTSLICES, dtype=float)
    for i in range(eicS.shape[0]):
        cPos = bestRTInd - int(peakbot_MRM.Config.RTSLICES/2.) + i
        if 0 <= cPos < len(rts):
            rtsS[i] = rts[cPos]
            eicS[i] = eic[cPos]
    if np.sum(eicS) > 0:
        eicS = eicS / np.max(eicS)
    return rtsS, eicS

def getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, peakType, rtStart, rtEnd):
    bestRTInd = np.argmin(np.array([abs(t - refRT) for t in rtsS]))        
    bestRTStartInd = -1
    bestRTEndInd = -1
    if peakType:
        bestRTStartInd = np.argmin(np.array([abs(t - rtStart) for t in rtsS]))
        bestRTEndInd   = np.argmin(np.array([abs(t - rtEnd) for t in rtsS]))
    else:
        bestRTStartInd = random.randint(0, int((peakbot_MRM.Config.RTSLICES-1)/2))
        bestRTEndInd = random.randint(int((peakbot_MRM.Config.RTSLICES-1)/2), peakbot_MRM.Config.RTSLICES-1)
    return bestRTInd, peakType, bestRTStartInd, bestRTEndInd, rtStart, rtEnd

def validateExperiment(expName, targetFile, curatedPeaks, samplesPath, modelFile, 
                       expDir = None, logDir = None, 
                       MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                       allowedMZOffset = 0.05,
                       plotSubstance = None, excludeSubstances = None, includeSubstances = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if plotSubstance is None:
        plotSubstances = []
    if excludeSubstances is None:
        excludeSubstances = []
        
    
    print("Validating experiment")
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
    print("\n")
    
    print("PeakBot configuration")
    print(peakbot_MRM.Config.getAsStringFancy())
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
        
    histAll = None
    metricsTable = {}

    substances               = peakbot_MRM.importTargets(targetFile, excludeSubstances = excludeSubstances, includeSubstances = includeSubstances)
    substances, integrations = peakbot_MRM.loadIntegrations(substances, curatedPeaks)
    substances, integrations = peakbot_MRM.loadChromatogramsTo(substances, integrations, samplesPath, expDir,
                                                               allowedMZOffset = allowedMZOffset,
                                                               MRMHeader = MRMHeader)


    print("Evaluating model (using predtrained model from '%s')"%(modelFile))
    offsetEIC = 0.2
    offsetRT1 = 0
    offsetRT2 = 0
    offsetRTMod = 1
    pbModelPred = peakbot_MRM.loadModel(modelFile, mode="predict", verbose = False)
    pbModelEval = peakbot_MRM.loadModel(modelFile, mode="training", verbose = False)
    for substance in tqdm.tqdm(integrations.keys()):
        if plotSubstance == "all" or substance in plotSubstance:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey = False, sharex = True)
            fig.set_size_inches(15, 8)

        temp = {"channel.int"  : np.zeros((len(integrations[substance]), peakbot_MRM.Config.RTSLICES), dtype=float),
                "channel.rts"  : np.zeros((len(integrations[substance]), peakbot_MRM.Config.RTSLICES), dtype=float),
                "inte.peak"    : np.zeros((len(integrations[substance]), peakbot_MRM.Config.NUMCLASSES), dtype=int),
                "inte.rtInds"  : np.zeros((len(integrations[substance]), 2), dtype=float),
                }
        agreement = np.zeros((4))
        
        for samplei, sample in enumerate(integrations[substance].keys()):
            assert len(integrations[substance][sample]["chrom"]) <= 1
            
            if len(integrations[substance][sample]["chrom"]) == 1:
                rts, eic = zip(*integrations[substance][sample]["chrom"][0][9])
                refRT = substances[substance]["RT"]
                
                ## standardize EIC
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                
                ## get integration results on standardized area
                bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                    getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    integrations[substance][sample]["foundPeak"], 
                                                    integrations[substance][sample]["rtstart"], 
                                                    integrations[substance][sample]["rtend"])
                
                temp["channel.int"][samplei, :] = eicS
                temp["inte.peak"][samplei,0] = 1 if peakType else 0
                temp["inte.peak"][samplei,1] = 1 if not peakType else 0
                temp["inte.rtInds"][samplei, 0] = bestRTStartInd
                temp["inte.rtInds"][samplei, 1] = bestRTEndInd
                temp["pred.peak"] = temp["inte.peak"]
                temp["pred.rtInds"] = temp["inte.rtInds"]
            
        ppeakTypes, prtStartInds, prtEndInds = peakbot_MRM.runPeakBot(temp, model = pbModelPred, verbose = False)
        metrics = peakbot_MRM.evaluatePeakBot(temp, model = pbModelEval, verbose = False)
        
        for samplei, sample in enumerate(integrations[substance].keys()):
            assert len(integrations[substance][sample]["chrom"]) <= 1
            
            if len(integrations[substance][sample]["chrom"]) == 1:
                rts, eic = zip(*integrations[substance][sample]["chrom"][0][9])
                refRT = substances[substance]["RT"]
                
                ## standardize EIC
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
            
                ## get integration results on standardized area
                bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                    getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    integrations[substance][sample]["foundPeak"], 
                                                    integrations[substance][sample]["rtstart"], 
                                                    integrations[substance][sample]["rtend"])
                    
                ## test if eic has detected signals
                
                ppeakType = np.argmax(ppeakTypes[samplei,:])
                ppeakType = ppeakType == 0
                prtStartInd = round(prtStartInds[samplei])
                prtEndInd = round(prtEndInds[samplei])
                prtStart = rtsS[min(peakbot_MRM.Config.RTSLICES-1, max(0, prtStartInd))]
                prtEnd = rtsS[min(peakbot_MRM.Config.RTSLICES-1, max(0, prtEndInd))]
                agreement[0] = agreement[0] + (1 if peakType and ppeakType else 0)
                agreement[1] = agreement[1] + (1 if peakType and not ppeakType else 0)
                agreement[2] = agreement[2] + (1 if not peakType and ppeakType else 0)
                agreement[3] = agreement[3] + (1 if not peakType and not ppeakType else 0)
                
                #print(substance, sample, "Predicted:", ppeakType, prtStartInd, prtEndInd, prtStart, prtEnd, "Integration:", peakType, bestRTStart, bestRTEnd, bestRTStartInd, bestRTEndInd)
                #print(metrics)
                if plotSubstance == "all" or substance in plotSubstance:
                    ## plot results
                    ## find correct axis to plot to 
                    ax = ax1
                    if peakType:
                        ax = ax1 if ppeakType else ax2
                    else:
                        ax = ax3 if ppeakType else ax4
                    
                    ## plot raw data
                    b = min(eic)
                    m = max([i-b for i in eic])
                    ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], [(e-b)/m+offsetEIC*samplei for e in eic], "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], [(e-b)/m+offsetEIC*samplei for e in eic], offsetEIC*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                    ax.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [offsetEIC*samplei, offsetEIC*samplei], "slategrey", linewidth=0.5, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    
                    ## add detected peak
                    if ppeakType:
                        ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if prtStart <= t <= prtEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=1, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if prtStart <= t <= prtEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], offsetEIC*samplei, facecolor='yellowgreen', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)

                    ## add integratin results
                    if peakType:            
                        ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if bestRTStart <= t <= bestRTEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=.5, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    
            
        ## log metrics
        TabLog().addData(substance, "Int.Peak, PrePeak", agreement[0])
        TabLog().addData(substance, "Int.Peak, PreNoPe", agreement[1])
        TabLog().addData(substance, "Int.NoPe, PrePeak", agreement[2])
        TabLog().addData(substance, "Int.NoPe, PreNoPe", agreement[3])
        TabLog().addData(substance, "Peakform", substances[substance]["PeakForm"])
        TabLog().addData(substance, "Rt shifts", substances[substance]["Rt shifts"])
        TabLog().addData(substance, "Note", substances[substance]["Note"])
        for k, v in metrics.items():
            TabLog().addData(substance, k, "%.4g"%(v), addNumToExistingKeys = True)
        
        if substance not in metricsTable.keys():
            metricsTable[substance] = {}
        metricsTable[substance] = {"peakCategoricalAccuracy": metrics["pred.peak_categorical_accuracy"], "AreaIOU": metrics["pred.rtInds_iou"]}
        
        if plotSubstance == "all" or substance in plotSubstance:
            ## add retention time of peak
            ax1.axvline(x = substances[substance]["RT"], zorder = 1E6, alpha = 0.1)
            ax2.axvline(x = substances[substance]["RT"], zorder = 1E6, alpha = 0.1)
            ax3.axvline(x = substances[substance]["RT"], zorder = 1E6, alpha = 0.1)
            ax4.axvline(x = substances[substance]["RT"], zorder = 1E6, alpha = 0.1)

            ## add title and scale accordingly
            ax1.set(xlabel = 'time (min)', ylabel = 'abundance')
            ax2.set(xlabel = 'time (min)', ylabel = 'abundance')
            ax3.set(xlabel = 'time (min)', ylabel = 'abundance')
            ax4.set(xlabel = 'time (min)', ylabel = 'abundance')
            ax1.set_title('Integration peak\nPrediciton peak', loc="left")
            ax2.set_title('Integration peak\nPrediction no peak', loc="left")
            ax3.set_title('Integration no peak\nPrediction peak', loc="left")
            ax4.set_title('Integration no peak\nPrediction no peak', loc="left")
            ax1.set_ylim(-0.2, len(integrations[substance].keys()) * offsetEIC + 1 + 0.2)
            ax2.set_ylim(-0.2, len(integrations[substance].keys()) * offsetEIC + 1 + 0.2)
            ax3.set_ylim(-0.2, len(integrations[substance].keys()) * offsetEIC + 1 + 0.2)
            ax4.set_ylim(-0.2, len(integrations[substance].keys()) * offsetEIC + 1 + 0.2,)
            fig.suptitle('%s\n%s\n%s'%(substance, substances[substance]["PeakForm"] + "; " + substances[substance]["Rt shifts"] + "; " + substances[substance]["Note"], "; ".join("%s: %.4g"%(k, v) for k, v in metrics.items())), fontsize=14)

            plt.tight_layout()
            fig.savefig(os.path.join(expDir, "SubstanceFigures","%s.png"%substance), dpi = 600)
            plt.close()
    print("\n")

    with open (os.path.join(expDir, "metricsTable.pickle"), "wb") as fout:
        pickle.dump(metricsTable, fout)
    with open (os.path.join(expDir, "metricsTable.pickle"), "rb") as fin:
        metricsTable = pickle.load(fin)

    for metric in ["peakCategoricalAccuracy", "AreaIOU"]:
        rows = [0]
        cols = [i for i in metricsTable.keys()]
        heatmapData = np.zeros((len(rows), len(cols)))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                heatmapData[i,j] = metricsTable[c][metric]

        fig, ax = plt.subplots()
        fig.set_size_inches(32, 8)
        
        if metric=="peakCategoricalAccuracy":
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
        #for i in range(len(rows)):
        #    for j in range(len(cols)):
        #        text = ax.text(j, i, "%.2f"%heatmapData[i, j], ha="center", va="center", color="w")
        plt.tight_layout()
        fig.savefig(os.path.join(expDir,"Metric_%s.png"%(metric)), dpi = 300)
        plt.close()


    TabLog().exportToFile(os.path.join(expDir, "results.tsv"))
    print("All calculations took %.1f seconds"%(toc("Overall process")))





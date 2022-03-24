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
from scipy.cluster.hierarchy import ward, single, complete, average, median, leaves_list
from scipy.spatial.distance import pdist
import pandas as pd
import random
random.seed(2021)
import os
import math


import PeakBotMRM
import PeakBotMRM.train
from PeakBotMRM.core import tic, toc, TabLog, extractStandardizedEIC, getInteRTIndsOnStandardizedEIC
print("\n")



def validateExperiment(expName, valDSs, modelFile, 
                       expDir = None, logDir = None, 
                       MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                       allowedMZOffset = 0.05,
                       plotSubstance = None, excludeSubstances = None, includeSubstances = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if excludeSubstances is None:
        excludeSubstances = []
        
    
    print("Validating experiment")
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
        substances               = PeakBotMRM.loadTargets(valDS["targetFile"], 
                                                          excludeSubstances = valDS["excludeSubstances"], 
                                                          includeSubstances = valDS["includeSubstances"])
        substances, integrations = PeakBotMRM.loadIntegrations(substances, valDS["curatedPeaks"])
        substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, valDS["samplesPath"],
                                                                allowedMZOffset = allowedMZOffset,
                                                                MRMHeader = MRMHeader)

        PeakBotMRM.train.investigatePeakMetrics(expDir, substances, integrations, expName = "%s"%(valDS["DSName"]))

        print("Evaluating model (using predtrained model from '%s')"%(modelFile))
        offsetEIC = 0.2
        offsetRT1 = 0
        offsetRT2 = 0
        offsetRTMod = 1
        pbModelPred = PeakBotMRM.loadModel(modelFile, mode="predict", verbose = False)
        pbModelEval = PeakBotMRM.loadModel(modelFile, mode="training", verbose = False)
        allSubstances = set()
        allSamples = set()
        
        subsN = set()
        sampN = set()
        for substance in integrations.keys():
            subsN.add(substance)
            for sample in integrations[substance]:
                sampN.add(sample)
        perInstanceResults = np.ones((len(subsN), len(sampN)), dtype=float)
        perInstanceResultsPD = []
        perInstanceResultsSubstances = list(subsN)
        perInstanceResultsSamples = list(sampN) 
        
        for substance in tqdm.tqdm(integrations.keys(), desc="  | .. comparing"):
            allSubstances.add(substance)
            if plotSubstance == "all" or substance in plotSubstance:
                fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3,4, sharey = "row", sharex = True, gridspec_kw = {'height_ratios':[2,1,1]})
                fig.set_size_inches(15, 16)

            temp = {"channel.int"  : np.zeros((len(integrations[substance]), PeakBotMRM.Config.RTSLICES), dtype=float),
                    "channel.rts"  : np.zeros((len(integrations[substance]), PeakBotMRM.Config.RTSLICES), dtype=float),
                    "inte.peak"    : np.zeros((len(integrations[substance]), PeakBotMRM.Config.NUMCLASSES), dtype=int),
                    "inte.rtInds"  : np.zeros((len(integrations[substance]), 2), dtype=float),
                    }
            truth = np.zeros((4))
            agreement = np.zeros((4))
            
            for samplei, sample in enumerate(integrations[substance].keys()):
                inte = integrations[substance][sample]
                allSamples.add(sample)
                assert len(inte["chrom"]) <= 1
                
                if len(inte["chrom"]) == 1:
                    rts = inte["chrom"][0][9]["rts"]
                    eic = inte["chrom"][0][9]["eic"]
                    refRT = substances[substance]["RT"]
                    
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                    
                    ## get integration results on standardized area
                    bestRTInd, isPeak, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                        getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                        inte["foundPeak"], 
                                                        inte["rtstart"], 
                                                        inte["rtend"])
                    
                    temp["channel.int"][samplei, :] = eicS
                    temp["inte.peak"][samplei,0] = 1 if isPeak else 0
                    temp["inte.peak"][samplei,1] = 1 if not isPeak else 0
                    temp["inte.rtInds"][samplei, 0] = bestRTStartInd
                    temp["inte.rtInds"][samplei, 1] = bestRTEndInd
                    temp["pred.peak"] = temp["inte.peak"]
                    temp["pred.rtInds"] = temp["inte.rtInds"]
                
            ppeakTypes, prtStartInds, prtEndInds = PeakBotMRM.runPeakBotMRM(temp, model = pbModelPred, verbose = False)
            metrics = PeakBotMRM.evaluatePeakBotMRM(temp, model = pbModelEval, verbose = False)
            
            for samplei, sample in enumerate(integrations[substance].keys()):
                inte = integrations[substance][sample] 
                assert len(inte["chrom"]) <= 1
                
                if len(inte["chrom"]) == 1:
                    rts = inte["chrom"][0][9]["rts"]
                    eic = inte["chrom"][0][9]["eic"]
                    refRT = substances[substance]["RT"]
                    
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                
                    ## get integration results on standardized area
                    bestRTInd, isPeak, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                        getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                        inte["foundPeak"], 
                                                        inte["rtstart"], 
                                                        inte["rtend"])
                        
                    ## test if eic has detected signals
                    pisPeak = ppeakTypes[samplei] == 0
                    prtStartInd = round(prtStartInds[samplei])
                    prtEndInd = round(prtEndInds[samplei])
                    prtStart = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, prtStartInd))]
                    prtEnd = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, prtEndInd))]
                    truth[0] = truth[0] + (1 if isPeak else 0)
                    truth[3] = truth[3] + (1 if not isPeak else 0)
                    agreement[0] = agreement[0] + (1 if isPeak and pisPeak else 0)
                    agreement[1] = agreement[1] + (1 if isPeak and not pisPeak else 0)
                    agreement[2] = agreement[2] + (1 if not isPeak and pisPeak else 0)
                    agreement[3] = agreement[3] + (1 if not isPeak and not pisPeak else 0)

                    inte["pred.rtstart"] = prtStart
                    inte["pred.rtend"] = prtEnd
                    inte["pred.foundPeak"] = pisPeak
                    if inte["pred.foundPeak"]:
                        inte["pred.area"] = PeakBotMRM.integrateArea(eic, rts, prtStart, prtEnd, method = "minbetweenborders")
                    else:
                        inte["pred.area"] = -1
                    if bestRTStart > 0:
                        inte["area."] = PeakBotMRM.integrateArea(eic, rts, bestRTStart, bestRTEnd, method = "minbetweenborders")
                    else:
                        inte["area."] = -1
                    
                    ## generate heatmap matrix
                    val = 0
                    if not isPeak and not pisPeak:
                        ## both (manual and prediction) report not a peak, 100% correct
                        val = 0
                    elif isPeak and not pisPeak:
                        ## manual integration reports a peak, but prediction does not, 100% incorrect
                        val = -1.00001
                    elif not isPeak and pisPeak:
                        ## manual integration reports not a peak, but prediction does, 100% incorrect
                        val = 1.00001
                    elif isPeak and pisPeak:
                        ## manual and prediction report a peak, x% correct
                        if inte["area."] < inte["pred.area"]:
                            val = min(1, inte["pred.area"] / inte["area."] - 1)
                        else:
                            val = -min(1, inte["area."] / inte["pred.area"] - 1)
                            
                    if val > 1.001 or val < -1.001:
                        print(substance, sample, isPeak, inte["area."], bestRTStart, bestRTEnd, pisPeak, inte["pred.area"], prtStart, prtEnd)
                    rowInd = perInstanceResultsSubstances.index(substance)
                    colInd = perInstanceResultsSamples.index(sample)
                    perInstanceResults[rowInd, colInd] = val
                    perInstanceResultsPD.append((substance, sample, val, isPeak, inte["area."], inte["area"], pisPeak, inte["pred.area"]))
                    
                    if plotSubstance == "all" or substance in plotSubstance:
                        ## plot results; find correct axis to plot to 
                        ax = ax1
                        axR = ax5
                        axS = ax9
                        if isPeak:
                            ax = ax1 if pisPeak else ax2
                            axR = ax5 if pisPeak else ax6
                            axS = ax9 if pisPeak else ax10
                        else:
                            ax = ax3 if pisPeak else ax4
                            axR = ax7 if pisPeak else ax8
                            axS = ax11 if pisPeak else ax12
                                            
                        ax.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [offsetEIC*samplei, offsetEIC*samplei], "slategrey", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        axR.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [0,0], "slategrey", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        axS.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [0,0], "slategrey", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        
                        ## plot raw, scaled data according to classification prediction and integration result
                        b = min(eic)
                        m = max([i-b for i in eic])
                        ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], [(e-b)/m+offsetEIC*samplei for e in eic], "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], [(e-b)/m+offsetEIC*samplei for e in eic], offsetEIC*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                        ## add detected peak
                        if pisPeak:
                            ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if prtStart <= t <= prtEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                            ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if prtStart <= t <= prtEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], offsetEIC*samplei, facecolor='yellowgreen', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                        ## add integration results
                        if isPeak:            
                            ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if bestRTStart <= t <= bestRTEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                                                    
                        ## plot raw data
                        axR.plot(rts, eic, "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ## add detected peak
                        if pisPeak:
                            axR.plot([t for t in rts if prtStart <= t <= prtEnd], [e for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ## add integration results
                        if isPeak:            
                            axR.plot([t for t in rts if bestRTStart <= t <= bestRTEnd], [e for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)

                        ## plot scaled data
                        ## add detected peak
                        minInt = min([e for e in eicS if e > 0])
                        maxInt = max([e-minInt for e in eicS])
                        axS.plot(rts, [(e-minInt)/maxInt for e in eic], "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        if pisPeak:
                            axS.plot([t for t in rts if prtStart <= t <= prtEnd], [(e-minInt)/maxInt for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        if isPeak:
                            axS.plot([t for t in rts if bestRTStart <= t <= bestRTEnd], [(e-minInt)/maxInt for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)

            
            if substance not in metricsTable.keys():
                metricsTable[substance] = {}
            metricsTable[substance] = {"CCA"          : metrics["pred.peak_categorical_accuracy"], 
                                       "MCC"          : metrics["pred.peak_MatthewsCorrelationCoefficient"],
                                       "RtMSE"        : metrics["pred.rtInds_MSE"], 
                                       "EICIOUPeaks"  : metrics["pred_EICIOUPeaks"],
                                       "Acc4Peaks"    : metrics["pred.peak_Acc4Peaks"],
                                       "Acc4NonPeaks" : metrics["pred.peak_Acc4NonPeaks"]}
            if substance in integrations.keys():
                intes = []
                intesD = []
                preds = []
                for sample in allSamples:
                    if sample in integrations[substance].keys():
                        intes.append(integrations[substance][sample]["area"])
                        intesD.append(integrations[substance][sample]["area."])
                        preds.append(integrations[substance][sample]["pred.area"])
                corr = np.corrcoef(intes, preds)[1,0]
                metricsTable[substance]["CorrIP"] = corr
                corr = np.corrcoef(intesD, preds)[1,0]
                metricsTable[substance]["CorrIdP"] = corr

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
            TabLog().addData(substance, "CorrIP", "%.4g"%(metricsTable[substance]["CorrIP"]), addNumToExistingKeys = True)
            TabLog().addData(substance, "CorrIdP", "%.4g"%(metricsTable[substance]["CorrIdP"]), addNumToExistingKeys = True)
            
            if plotSubstance == "all" or substance in plotSubstance:
                ## add retention time of peak
                for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
                    ax.axvline(x = substances[substance]["RT"], zorder = 1E6, alpha = 0.2)

                ## add title and scale accordingly
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.set(xlabel = 'time (min)', ylabel = 'rel. abundance')
                    ax.set_ylim(-0.2, len(integrations[substance].keys()) * offsetEIC + 1 + 0.2)
                for ax in [ax5, ax6, ax7, ax8]:
                    ax.set(xlabel = 'time (min)', ylabel = 'abundance')
                for ax in [ax9, ax10, ax11, ax12]:
                    ax.set_ylim(-0.1, 1.1)
                total = agreement[0] + agreement[1] + agreement[2] + agreement[3]
                ax1.set_title('Integration peak\nPrediciton peak\n%d (%.1f%%, integration %.1f%%)'%(agreement[0], agreement[0]/total*100, truth[0]/total*100), loc="left")
                ax2.set_title('Integration peak\nPrediction no peak\n%d (%.1f%%)'%(agreement[1], agreement[1]/total*100), loc="left")
                ax3.set_title('Integration no peak\nPrediction peak\n%d (%.1f%%)'%(agreement[2], agreement[2]/total*100), loc="left")
                ax4.set_title('Integration no peak\nPrediction no peak\n%d (%.1f%%, integration %.1f%%)'%(agreement[3], agreement[3]/total*100, truth[3]/total*100), loc="left")
                fig.suptitle('%s\n%s\n%s\nGreen EIC and area: prediction; black EIC: manual integration; grey EIC: standardized EIC; light grey EIC: raw data'%(substance, substances[substance]["PeakForm"] + "; " + substances[substance]["Rt shifts"] + "; " + substances[substance]["Note"], "; ".join("%s: %.4g"%(k, v) for k, v in metricsTable[substance].items())), fontsize=14)

                plt.tight_layout()
                fig.savefig(os.path.join(expDir, "SubstanceFigures","%s_%s.png"%(valDS["DSName"], substance)), dpi = 600)
                plt.close(fig)
        print("\n")

        allSubstances = list(allSubstances)
        allSamples = list(allSamples)
        with open(os.path.join(expDir, "%s_results_IntePred.tsv"%(valDS["DSName"])), "w") as fout:
            fout.write("Sample")
            for substance in allSubstances:
                fout.write("\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(substance, "","","", "","",""))
            fout.write("\n")

            fout.write("")
            for substance in allSubstances:
                fout.write("\tInt.Start\tInt.End\tInt.Area\tInt.Area.\tPred.Start\tPred.End\tPred.Area")
            fout.write("\n")

            for sample in allSamples:
                fout.write(sample)
                for substance in allSubstances:
                    if substance in integrations.keys() and sample in integrations[substance].keys() and len(integrations[substance][sample]["chrom"]) == 1:
                        temp = integrations[substance][sample]
                        ## Integration
                        if temp["area"] > 0:
                            fout.write("\t%.3f\t%.3f\t%d\t%.3f"%(temp["rtstart"], temp["rtend"], temp["area"], temp["area."]))
                        else:
                            fout.write("\t\t\t\t")
                        ## Prediction
                        if temp["pred.foundPeak"]:
                            fout.write("\t%.3f\t%.3f\t%.3f"%(temp["pred.rtstart"] if not np.isnan(temp["pred.rtstart"]) else -1, 
                                                            temp["pred.rtend"] if not np.isnan(temp["pred.rtend"]) else -1, 
                                                            temp["pred.area"] if not np.isnan(temp["pred.area"]) else -1))
                        else:
                            fout.write("\t\t\t")
                    else:
                        fout.write("\t\t\t\t\t\t\t")
                fout.write("\n")

        with open (os.path.join(expDir, "%s_metricsTable.pickle"%(valDS["DSName"])), "wb") as fout:
            pickle.dump(metricsTable, fout)
        with open (os.path.join(expDir, "%s_metricsTable.pickle"%(valDS["DSName"])), "rb") as fin:
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
            
            if False: 
                plt.colorbar(im)
                ax.set_xticks(np.arange(len(cols)))
                ax.set_yticks(np.arange(len([rows[i]])))
                ax.set_xticklabels(cols)
                ax.set_yticklabels([rows[i]])
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
                plt.tight_layout()
                fig.savefig(os.path.join(expDir,"%s_Metric_%s.png"%(valDS["DSName"], metric)), dpi = 300)
                plt.close()
            
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
        
        with open(os.path.join(expDir, "%s_pd_individualInstances.pickle"%(valDS["DSName"])), "wb") as fout:
            pickle.dump((perInstanceResults, perInstanceResultsPD, perInstanceResultsSamples, perInstanceResultsSubstances), fout)

        #with open("./R100140_METAB02/pd_individualInstances.pickle", "rb") as fin:
        #    perInstanceResults, perInstanceResultsPD, perInstanceResultsSamples, perInstanceResultsSubstances = pickle.load(fin)
        
        from scipy.cluster.hierarchy import ward, leaves_list
        from scipy.spatial.distance import pdist
        
        ordRow = leaves_list(ward(pdist(perInstanceResults)))
        ordCol = leaves_list(ward(pdist(np.transpose(perInstanceResults))))
        
        df = pd.DataFrame(perInstanceResultsPD, columns = ["substance", "sample", "value", "manualPeak", "manualAreaDOT", "manualArea", "PBPeak", "PBArea"])
        df["substance"] = df["substance"].astype("category")
        df["substance"].cat.reorder_categories([perInstanceResultsSubstances[i] for i in ordRow])
        df["sample"] = df["sample"].astype("category")
        df["sample"].cat.reorder_categories([perInstanceResultsSamples[i] for i in ordCol])
        
        ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
        plot = (p9.ggplot(df, p9.aes('np.log2(manualArea)', 'np.log2(manualAreaDOT)'))
        + p9.geom_point(alpha=0.1)
        + p9.geom_smooth(method='lm')
        + p9.ggtitle("Comparison of provided manual integration results and PeakBot integration based on the manual integration peak borders")
        )
        p9.options.figure_size = (10,10)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_comparisonManualIntegration1.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
        
        ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
        plot = (p9.ggplot(df, p9.aes('np.log2(manualArea)', 'np.log2(manualAreaDOT / manualArea)'))
        + p9.geom_point(alpha=0.1)
        + p9.geom_smooth(method='lm')
        + p9.ggtitle("Comparison of provided manual integration results and PeakBot integration based on the manual integration peak borders")
        )
        p9.options.figure_size = (10,10)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_comparisonManualIntegration2.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
        
        
        
        ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
        plot = (p9.ggplot(df, p9.aes('np.log2(manualAreaDOT)', 'np.log2(PBArea)'))
        + p9.geom_point(alpha=0.1)
        + p9.geom_smooth(method='lm')
        + p9.ggtitle("Comparison of manual integration with the PeakBot algorithm and PeakBot's predicted borders")
        )
        p9.options.figure_size = (10,10)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_comparisonIntegration1.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
        
        ## show correlation between areas integrated with XXX method and PeakBot's method using the manual integration peak borders
        plot = (p9.ggplot(df, p9.aes('np.log2(manualAreaDOT)', 'np.log2(manualAreaDOT / PBArea)'))
        + p9.geom_point(alpha=0.1)
        + p9.geom_smooth(method='lm')
        + p9.ggtitle("Comparison of manual integration with the PeakBot algorithm and PeakBot's predicted borders")
        )
        p9.options.figure_size = (10,10)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_comparisonIntegration2.png"%(valDS["DSName"])), width=10, height=10, dpi=300, verbose=False)
        
        
        
        ordRow = leaves_list(median(pdist(perInstanceResults, 'euclidean')))
        ordCol = leaves_list(median(pdist(np.transpose(perInstanceResults), 'euclidean')))
        
        df = pd.DataFrame(perInstanceResultsPD, columns = ["substance", "sample", "value", "manualPeak", "manualArea", "manualArea.", "PBPeak", "PBArea"])
        df = df.assign(sampleHC = pd.Categorical(df["sample"], categories = [perInstanceResultsSamples[i] for i in ordCol]))
        df = df.assign(substanceHC = pd.Categorical(df["substance"], categories = [perInstanceResultsSubstances[i] for i in ordRow]))
        df["value"] = round(df["value"], 5)

        plot = (p9.ggplot(df, p9.aes('substanceHC', 'sampleHC', fill='value'))
            + p9.geom_tile(p9.aes(width=.95, height=.95))
            + p9.geom_text(p9.aes(label='value'), size=1)
            + p9.theme(axis_text_x = p9.element_text(rotation=45, hjust=1))
            + p9.scales.scale_fill_gradientn(["Firebrick", "#440154FF", "#21908CFF", "Ghostwhite", "Ghostwhite", "Ghostwhite", "#21908CFF", "#FDE725FF", "Orange"], [(i+1.001)/2.002 for i in [-1.001, -1, -0.101, -0.1, 0, 0.1, 0.101, 1, 1.001]])
            + p9.ggtitle(expName + ": Heatmap of predicted and manually derived integration results.\n0 (white) indicates the perfect agreement (manual and prediction for peak/nopeak agree and identical peak areas (+/- 10%) if peaks were detected)\n1.001 (red) indicates a predicted peak but nopeak in the manual integration, while -1.001 (orange) indicates a nopeak in the prediction but a manually integrated peak\ncolors between -1 and 1 indicate the increase (positive) or decrease (negative) of the abundance difference relative manually integrated peak area (in %)\n" + "PBPeak & GTPeak %d (%.1f%%); PBNoPeak & GTNoPeak %d (%.1f%%); PBPeak & GTNoPeak %d (%.1f%%); PBNoPeak & GTPeak %d (%.1f%%)\n"%(sum(df["PBPeak"] & df["manualPeak"]), sum(df["PBPeak"] & df["manualPeak"]) / df.shape[0] * 100, sum(~df["PBPeak"] & ~df["manualPeak"]), sum(~df["PBPeak"] & ~df["manualPeak"])/ df.shape[0] * 100, sum(df["PBPeak"] & ~df["manualPeak"]), sum(df["PBPeak"] & ~df["manualPeak"]) / df.shape[0] * 100, sum(~df["PBPeak"] & df["manualPeak"]), sum(~df["PBPeak"] & df["manualPeak"]) / df.shape[0] * 100) + str(round(df[df["PBPeak"] & df["manualPeak"]]["value"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),2,)).replace("\n", "; ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " "))
            )
        p9.options.figure_size = (50, 20)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_perInstanceResults.pdf"%(valDS["DSName"])), width=50, height=20, limitsize=False, verbose=False)
        
    TabLog().exportToFile(os.path.join(expDir, "results.tsv"))
    print("All calculations took %.1f seconds"%(toc("Overall process")))





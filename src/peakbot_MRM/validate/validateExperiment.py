




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
                       plotSubstance = None):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if plotSubstance is None:
        plotSubstances = []
        
    
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


    ## load targets
    print("Loading targets from file '%s'"%(targetFile))
    headers, substances = readTSVFile(targetFile, header = True, delimiter = "\t", convertToMinIfPossible = True, getRowsAsDicts = True)
    substances = dict((substance["Name"], {"Name"     : substance["Name"].replace(" (ISTD)", ""),
                                        "Q1"       : substance["Precursor Ion"],
                                        "Q3"       : substance["Product Ion"],
                                        "RT"       : substance["RT"],
                                        "PeakForm" : substance["PeakForm"], 
                                        "Rt shifts": substance["RT shifts"],
                                        "Note"     : substance["Note"],
                                        "Pola"     : substance["Ion Polarity"],
                                        "ColE"     : None}) for substance in substances) # TODO add collision energy here for selection of correct channel
                ##TODO include collisionEnergy here
    print("  | .. loaded %d substances"%(len(substances)))
    print("  | .. of these %d have RT shifts"%(sum((1 if substance["Rt shifts"]!="" else 0 for substance in substances.values()))))
    print("  | .. of these %d have abnormal peak forms"%(sum((1 if substance["PeakForm"]!="" else 0 for substance in substances.values()))))
    print("\n")
    # targets: [{'Name': 'Valine', 'Q1': 176.0, 'Q3': 116.0, 'RT': 1.427}, ...]



    ## load integrations
    print("Loading integrations from file '%s'"%(curatedPeaks))
    headers, temp = parseTSVMultiLineHeader(curatedPeaks, headerRowCount=2, delimiter = ",", commentChar = "#", headerCombineChar = "$")
    headers = dict((k.replace(" (ISTD)", ""), v) for k,v in headers.items())
    foo = set([head[:head.find("$")] for head in headers if not head.startswith("Sample$")])
    print("  | .. Not using substances ", end="")
    for substance in substances.values():
        if substance["Name"] not in foo:
            print(substance["Name"], ", ", end="")
    print("as these are not in the integration matrix")
    foo = dict((k, v) for k, v in substances.items() if k in foo)
    print("  | .. restricting substances from %d to %d (overlap of substances and integration results)"%(len(substances), len(foo)))
    substances = foo

    ## process integrations
    integrations = {}
    integratedSamples = set()
    totalIntegrations = 0
    foundPeaks = 0
    foundNoPeaks = 0
    for substance in [substance["Name"] for substance in substances.values()]:
        integrations[substance] = {}
        for intei, inte in enumerate(temp):
            area = inte[headers["%s$Area"%(substance)]]
            if area == "" or float(area) == 0:
                integrations[substance][inte[headers["Sample$Name"]]] = {"foundPeak": False,
                                                                         "rtstart"  : -1, 
                                                                         "rtend"    : -1, 
                                                                         "area"     : -1,
                                                                         "chrom"    : [],}
                foundNoPeaks += 1
            else:
                integrations[substance][inte[headers["Sample$Name"]]] = {"foundPeak": True,
                                                                         "rtstart"  : float(inte[headers["%s$Int. Start"%(substance)]]), 
                                                                         "rtend"    : float(inte[headers["%s$Int. End"  %(substance)]]), 
                                                                         "area"     : float(inte[headers["%s$Area"      %(substance)]]),
                                                                         "chrom"    : [],}
                foundPeaks += 1
            integratedSamples.add(inte[headers["Sample$Name"]])
            totalIntegrations += 1
    print("  | .. parsed %d integrations from %d substances and %d samples"%(totalIntegrations, len(substances), len(integratedSamples)))
    print("  | .. there are %d areas and %d no peaks"%(foundPeaks, foundNoPeaks))
    print("\n")
    # integrations [['Pyridinedicarboxylic acid Results', 'R100140_METAB02_MCC025_CAL1_20200306', '14.731', '14.731', '0'], ...]



    ## load chromatograms
    tic("procChroms")
    print("Processing chromatograms")
    samples = [os.path.join(samplesPath, f) for f in os.listdir(samplesPath) if os.path.isfile(os.path.join(samplesPath, f)) and f.lower().endswith(".mzml")]
    usedSamples = set()
    if os.path.isfile(os.path.join(expDir, "integrations.pickle")):
        with open(os.path.join(expDir, "integrations.pickle"), "rb") as fin:
            integrations, referencePeaks, noReferencePeaks, usedSamples = pickle.load(fin)
            print("  | .. Imported integrations from pickle file '%s/integrations.pickle'"%(expDir))
    else:
        print("  | .. This might take a couple of minutes as all samples/integrations/channels/etc. need to be compared and the current implementation are 4 sub-for-loops")
        for sample in tqdm.tqdm(samples):
            sampleName = os.path.basename(sample)
            sampleName = sampleName[:sampleName.rfind(".")]
            if sampleName in integratedSamples:# and sampleName == "R100140_METAB02_MCC025_105713_20200306":
                usedSamples.add(sampleName)

                foundTargets = []
                unusedChannels = []
                run = pymzml.run.Reader(sample, skip_chromatogram = False)
                
                ## get channels from the chromatogram
                allChannels = []
                for i, entry in enumerate(run):
                    if isinstance(entry, pymzml.spec.Chromatogram) and entry.ID.startswith("- SRM"):
                        m = re.match(MRMHeader, entry.ID)
                        Q1, Q3, rtstart, rtend = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))

                        polarity = None
                        if entry.get_element_by_name("negative scan") is not None:
                            polarity = "negative"
                        elif entry.get_element_by_name("positive scan") is not None:
                            polarity = "positive"

                        collisionEnergy = None
                        if entry.get_element_by_name("collision energy") is not None:
                            collisionEnergy = entry.get_element_by_name("collision energy").get("value", default=None)
                            if collisionEnergy is not None:
                                collisionEnergy = float(collisionEnergy)

                        collisionType = None
                        if entry.get_element_by_name("collision-induced dissociation") is not None:
                            collisionType = "collision-induced dissociation"

                        chrom = [(time, intensity) for time, intensity in entry.peaks()]

                        allChannels.append([Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entry.ID, chrom])

                ## merge channels with integration results for this sample
                for i, (Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID, chrom) in enumerate(allChannels):
                    usedChannel = []
                    useChannel = True
                    ## test if channel is unique ## TODO include collisionEnergy here as well
                    for bi, (bq1, bq3, brtstart, brtend, bpolarity, bcollisionEnergy, bcollisionType, bentryID, bchrom) in enumerate(allChannels):
                        if i != bi:
                            if abs(Q1 - bq1) <= allowedMZOffset and abs(Q3 - bq3) <= allowedMZOffset and \
                                polarity == bpolarity and collisionType == bcollisionType:# TODO include collisionEnergy test here and collisionEnergy == bcollisionEnergy:
                                useChannel = False
                                unusedChannels.append(entryID)
                    
                    ## use channel if it is unique and find the integrated substance(s) for it
                    if useChannel:
                        for substance in substances.values(): ## TODO include collisionEnergy check here
                            if abs(substance["Q1"] - Q1) < allowedMZOffset and abs(substance["Q3"] - Q3) <= allowedMZOffset and rtstart <= substance["RT"] <= rtend:
                                if substance["Name"] in integrations.keys() and sampleName in integrations[substance["Name"]].keys():
                                    foundTargets.append([substance, entry, integrations[substance["Name"]][sampleName]])
                                    usedChannel.append(substance)
                                    integrations[substance["Name"]][sampleName]["chrom"].append(["%s (%s mode, %s with %.1f energy)"%(entryID, polarity, collisionType, collisionEnergy), 
                                                                                                Q1, Q3, rtstart, rtend, polarity, collisionEnergy, collisionType, entryID, chrom])
        
        ## remove all integrations with more than one scanEvent
        referencePeaks = 0
        noReferencePeaks = 0
        for substance in integrations.keys():
            for sample in integrations[substance].keys():
                if len(integrations[substance][sample]["chrom"]) == 1:
                    referencePeaks += 1
                else:
                    noReferencePeaks += 1
                    integrations[substance][sample]["chrom"].clear()

        with open (os.path.join(expDir, "integrations.pickle"), "wb") as fout:
            pickle.dump((integrations, referencePeaks, noReferencePeaks, usedSamples), fout)
            print("  | .. Stored integrations to '%s/integrations.pickle'"%expDir)
        
    print("  | .. There are %d training peaks and %d no peaks"%(referencePeaks, noReferencePeaks))
    print("  | .. Using %d samples "%(len(usedSamples)))
    remSubstancesChannelProblems = []
    for substance in integrations.keys():
        foundOnce = False
        for sample in integrations[substance].keys():
            if len(integrations[substance][sample]["chrom"]) > 1:
                remSubstancesChannelProblems.append(substance)
                break
            elif len(integrations[substance][sample]["chrom"]) == 1:
                foundOnce = True
        if not foundOnce:
            remSubstancesChannelProblems.append(substance)
    if len(remSubstancesChannelProblems):
        print("  | .. %d substances (%s) were not found as the channel selection was ambiguous"%(len(remSubstancesChannelProblems), ", ".join(sorted(remSubstancesChannelProblems))))
        print("  | .. These will not be used further")
        for r in remSubstancesChannelProblems:
            del integrations[r]
    print("  | .. took %.1f seconds"%(toc("procChroms")))
    print("\n")


    print("Evaluating model")
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

        temp = {"channel.int": np.zeros((len(integrations[substance].keys()), peakbot_MRM.Config.RTSLICES), dtype=float),
                "inte.peak"  : np.zeros((len(integrations[substance].keys()), peakbot_MRM.Config.NUMCLASSES), dtype=int),
                "inte.rtInds": np.zeros((len(integrations[substance].keys()), 2), dtype=float),}
        curI = 0
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
                
                if np.sum(eicS) > 0:                    
                    ## export to peakbot_MRM dictionary for prediction
                    temp["channel.int"][curI, :] = eicS
                
                temp["inte.peak"][curI,0] = 1 if peakType else 0
                temp["inte.peak"][curI,1] = 1 if not peakType else 0
                temp["inte.rtInds"][curI, 0] = bestRTStartInd
                temp["inte.rtInds"][curI, 1] = bestRTEndInd
                curI = curI + 1
                
        if curI > 0:
            temp["channel.int"] = temp["channel.int"][0:curI, :]
            temp["inte.peak"] = temp["inte.peak"][0:curI, :]
            temp["inte.rtInds"] = temp["inte.rtInds"][0:curI, :]
            
            ppeakTypes, prtStartInds, prtEndInds = peakbot_MRM.runPeakBot(temp, model = pbModelPred, verbose = False)
            metrics = peakbot_MRM.evaluatePeakBot(temp, model = pbModelEval, verbose = False)
            
            curI = 0
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
                    
                    ppeakType = np.argmax(ppeakTypes[curI,:])
                    ppeakType = ppeakType == 0
                    prtStartInd = round(prtStartInds[curI])
                    prtEndInd = round(prtEndInds[curI])
                    prtStart = rtsS[min(peakbot_MRM.Config.RTSLICES-1, max(0, prtStartInd))]
                    prtEnd = rtsS[min(peakbot_MRM.Config.RTSLICES-1, max(0, prtEndInd))]
                    agreement[0] = agreement[0] + (1 if peakType and ppeakType else 0)
                    agreement[1] = agreement[1] + (1 if peakType and not ppeakType else 0)
                    agreement[2] = agreement[2] + (1 if not peakType and ppeakType else 0)
                    agreement[3] = agreement[3] + (1 if not peakType and not ppeakType else 0)
                    
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
                    
                    curI = curI + 1
            
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
        else:
            print(substance)
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





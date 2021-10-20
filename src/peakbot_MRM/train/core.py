
from peakbot_MRM.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary
import peakbot_MRM
from peakbot_MRM.core import readTSVFile, parseTSVMultiLineHeader
from peakbot_MRM.validate import extractStandardizedEIC, getInteRTIndsOnStandardizedEIC

import os
import pathlib
import tempfile
import matplotlib.pyplot as plt
import matplotlib as mpl
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

import pymzml



def shuffleResultsSampleNames(exportPath, instancePrefix=None,
                              tempFileName="bqu40mcb25473zfhbgwh22534", verbose=False):

    if instancePrefix is None:
        instancePrefix = peakbot_MRM.Config.INSTANCEPREFIX

    tic("shuffling")
    if verbose:
        print("Shuffling the test instances (batch name shuffling)")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose:
        print("  | .. there are %d files" % (len(files)))

    unused = [t for t in range(len(files))]
    random.shuffle(unused)
    for i in range(len(files)):
        j = unused[0]
        del unused[0]

        os.rename(files[i], os.path.join(pathlib.Path(files[i]).parent.resolve(), "%s%d.pickle" % (tempFileName, i)))

    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    for i in range(len(files)):
        os.rename(files[i], files[i].replace(tempFileName, instancePrefix))

    if verbose:
        print("  | .. took %.1f seconds" % toc("shuffling"))
        print("")


def shuffleResults(exportPath, steps=1E5, samplesToExchange=50,
                   instancePrefix=None, verbose=False):

    if instancePrefix is None:
        instancePrefix = peakbot_MRM.Config.INSTANCEPREFIX

    tic("shuffling")
    if verbose:
        print("Shuffling the test instances (inter-batch shuffling)")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose:
        print("  | .. there are %d files" % (len(files)))

    with tqdm.tqdm(total=steps, desc="  | .. shuffling", disable=not verbose) as t:
        while steps > 0:
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

            cExchange = min(min(samplesA, samplesB), samplesToExchange)

            beginA = random.randint(0, samplesA - cExchange)
            beginB = random.randint(0, samplesB - cExchange)

            for k in a.keys():
                if isinstance(a[k], np.ndarray) and len(a[k].shape) == 1:
                    temp = a[k][beginA:(beginA + cExchange)]
                    a[k][beginA:(beginA + cExchange)] = b[k][beginB:(beginB + cExchange)]
                    b[k][beginB:(beginB + cExchange)] = temp

                elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 2:
                    temp = a[k][beginA:(beginA + cExchange), :]
                    a[k][beginA:(beginA + cExchange),:] = b[k][beginB:(beginB + cExchange), :]
                    b[k][beginB:(beginB + cExchange), :] = temp

                elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 3:
                    temp = a[k][beginA:(beginA + cExchange), :, :]
                    a[k][beginA:(beginA + cExchange), :,:] = b[k][beginB:(beginB + cExchange), :, :]
                    b[k][beginB:(beginB + cExchange), :, :] = temp

                elif isinstance(a[k], np.ndarray) and len(a[k].shape) == 4:
                    temp = a[k][beginA:(beginA + cExchange), :, :, :]
                    a[k][beginA:(beginA + cExchange), :, :,:] = b[k][beginB:(beginB + cExchange), :, :, :]
                    b[k][beginB:(beginB + cExchange), :, :, :] = temp

                elif isinstance(a[k], list):
                    temp = a[k][beginA:(beginA + cExchange)]
                    a[k][beginA:(beginA + cExchange)] = b[k][beginB:(beginB + cExchange)]
                    b[k][beginB:(beginB + cExchange)] = temp
                
                else:
                    assert False, "Unknown key in shuffling, aborting"

            assert samplesA == a["channel.rt"].shape[0] and samplesB == b["channel.rt"].shape[0]

            with open(filea, "wb") as temp:
                pickle.dump(a, temp)
            with open(fileb, "wb") as temp:
                pickle.dump(b, temp)

            steps = steps - 1
            t.update()

    if verbose:
        print("  | .. took %.1f seconds" % toc("shuffling"))
        print("")


def splitDSinto(path, newDS1Path, newDS2Path = None, ratioDS1 = 0.3, instancePrefix = None, tempFileName = "bqu40mcb25473zfhbgwh22534", copy=False, verbose = False):

    assert 0 <= ratioDS1 <= 1, "parameter ratioDS1 must be 0 <= ratioDS1 <= 1"
    
    pathlib.Path(newDS1Path).mkdir(parents=True, exist_ok=True) 
    if newDS2Path is not None:
        pathlib.Path(newDS2Path).mkdir(parents=True, exist_ok=True) 

    if instancePrefix is None:
        instancePrefix = peakbot_MRM.Config.INSTANCEPREFIX

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


def trainPeakBotMRMModel(expName, targetFile, curatedPeaks, samplesPath, modelFile, expDir = None, logDir = None, historyObject = None,
                         MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                         allowedMZOffset = 0.05, drawRawData = False,
                         addRandomNoise = True, maxRandFactor = 0.2, shiftRTs = True, maxShift = 0.25):
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if historyObject is None:
        historyObject = os.path.join(expDir, "History.pandas.pickle")
        
    

        
    
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

    
    print("PeakBot configuration")
    print(peakbot_MRM.Config.getAsStringFancy())
    print("\n")






















    ## administrative
    tic("Overall process")
    try: 
        os.remove(historyObject)
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
    totalTraningPeaks = 0
    totalTraningNoPeaks = 0
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



    if drawRawData:
        tic("drawEICs")
        offset = 0.2
        trainingsInstances = []
        print("Exporting illustrations of data")
        for substance in tqdm.tqdm(integrations.keys()):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=False)
            fig.set_size_inches(15, 8)

            nPeaks = 0
            nNoPeaks = 0
            for samplei, sample in enumerate(integrations[substance].keys()):
                assert len(integrations[substance][sample]["chrom"]) <= 1
                
                if len(integrations[substance][sample]["chrom"]) == 1:
                    foundPeak, rtStart, rtEnd, area = integrations[substance][sample]["foundPeak"], integrations[substance][sample]["rtstart"], integrations[substance][sample]["rtend"], integrations[substance][sample]["area"]
                    inte = integrations[substance][sample]["chrom"][0]
                    chrom = inte[9]
                    
                    if foundPeak:
                        ax1.plot([t[0] for t in chrom if t[0]<=rtStart], [t[1] for t in chrom if t[0]<=rtStart], label=sample, linewidth=.5, alpha=.3, color="slategrey")
                        ax1.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [t[1] for t in chrom if rtStart<=t[0]<=rtEnd], label=sample, linewidth=2, alpha=.3, color="firebrick")
                        ax1.plot([t[0] for t in chrom if rtEnd<=t[0]], [t[1] for t in chrom if rtEnd<=t[0]], label=sample, linewidth=.5, alpha=.3, color="slategrey")

                        temp = [t[1] for t in chrom if rtStart<=t[0]<=rtEnd]
                        if len(temp) > 0:
                            ax2.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [t[1] for t in chrom if rtStart<=t[0]<=rtEnd], label=sample, linewidth=.5, alpha=.3, color="firebrick")
                            b = min(temp)
                            m = max([i-b for i in temp])
                            #ax3.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [(t[1]-b)/m for t in chrom if rtStart<=t[0]<=rtEnd], label=sample, linewidth=.5, alpha=.3, color="firebrick")
                        
                            ax3.plot([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [(t[1]-b)/m+offset*samplei for t in chrom if rtStart<=t[0]<=rtEnd], "k", linewidth=.5, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                            ax3.fill_between([t[0] for t in chrom if rtStart<=t[0]<=rtEnd], [(t[1]-b)/m+offset*samplei for t in chrom if rtStart<=t[0]<=rtEnd], offset*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                        nPeaks +=1
                    else:
                        nNoPeaks += 1

                        ax4.plot([t[0] for t in chrom], [t[1] for t in chrom], label=sample, linewidth=.5, alpha=.3, color="slategrey")

                        temp = [t[1] for t in chrom]
                        b = min(temp)
                        m = max([i-b for i in temp])
                        ax5.plot([t[0] for t in chrom], [(t[1]-b)/m+offset*samplei for t in chrom], 'k', linewidth=.5, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ax5.fill_between([t[0] for t in chrom], [(t[1]-b)/m+offset*samplei for t in chrom], offset*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)

            ax1.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax2.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax3.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax4.axvline(x=substances[substance]["RT"], zorder=1E6)
            ax5.axvline(x=substances[substance]["RT"], zorder=1E6)

            ax1.set(xlabel='time (min)', ylabel='abundance', title='Peaks\nRaw')
            ax2.set(xlabel='time (min)', ylabel='abundance', title='%s (%d peaks, %d no peaks)\nZoomed'%(substance, nPeaks, nNoPeaks))
            ax3.set(xlabel='time (min)', ylabel='abundance', title='\nNormalized')
            ax4.set(xlabel='time (min)', ylabel='abundance', title='No peaks\nRaw')
            ax5.set(xlabel='time (min)', ylabel='abundance', title='\nNormalized')

            ax3.set_ylim(-0.2, len(integrations[substance].keys())*offset+1+0.2)
            ax5.set_ylim(-0.2, len(integrations[substance].keys())*offset+1+0.2)
            
            plt.tight_layout()
            fig.savefig(os.path.join(expDir, "%s.png"%substance), dpi=300)
            plt.close()
        print("  | .. took %.1f seconds"%toc("drawEICs"))
        print("\n")



    ## export for training and detection
    instanceDirObj = tempfile.TemporaryDirectory()
    instanceDir = instanceDirObj.name
    temp = None
    cur = 0
    curI = 0
    ## iterate all samples and substances
    print("Exporting instances")
    if addRandomNoise:
        print("  | .. Random noise will be added. The range of the randomly generated factors is %.3f - %.3f"%(1 - maxRandFactor, 1 + maxRandFactor))
    if shiftRTs:
        print("  | .. Random RT shifts will be added. The range is -%.3f - %.3f minutes"%(maxShift, maxShift))
    for substance in tqdm.tqdm(integrations.keys()):
        for sample in integrations[substance].keys():
            if len(integrations[substance][sample]["chrom"]) == 1:
                
                inte = integrations[substance][sample]["chrom"][0]
                rts = [t[0] for t in inte[9]]
                eic = [t[1] for t in inte[9]]
                refRT = substances[substance]["RT"]
                artificialRTShift = np.random.rand(1) * 2 * maxShift - maxShift
                
                ## standardize EIC
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT + artificialRTShift)
                
                ## get integration results on standardized area
                bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                    getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    integrations[substance][sample]["foundPeak"], 
                                                    integrations[substance][sample]["rtstart"], 
                                                    integrations[substance][sample]["rtend"])
                
                ## add random noise
                if addRandomNoise:
                    eicS = eicS * (1 + np.random.rand(eicS.shape[0]) * 2 * maxRandFactor - maxRandFactor)
                    eicS = eicS / np.max(eicS)
                
                ## test if eic has detected signals
                if np.sum(eicS) > 0:
                    ## add instance to training data
                    if temp is None:
                        temp = {"channel.rt"        : np.zeros((peakbot_MRM.Config.BATCHSIZE, peakbot_MRM.Config.RTSLICES), dtype=float),
                                "channel.int"       : np.zeros((peakbot_MRM.Config.BATCHSIZE, peakbot_MRM.Config.RTSLICES), dtype=float),
                                "inte.peak"         : np.zeros((peakbot_MRM.Config.BATCHSIZE, peakbot_MRM.Config.NUMCLASSES), dtype=int),
                                "inte.rtStart"      : np.zeros((peakbot_MRM.Config.BATCHSIZE), dtype=float),
                                "inte.rtEnd"        : np.zeros((peakbot_MRM.Config.BATCHSIZE), dtype=float),
                                "inte.rtInds"       : np.zeros((peakbot_MRM.Config.BATCHSIZE, 2), dtype=float),
                                "inte.area"         : np.zeros((peakbot_MRM.Config.BATCHSIZE), dtype=float),
                                "ref.substanceName" : ["" for i in range(peakbot_MRM.Config.BATCHSIZE)],
                                "ref.sample"        : ["" for i in range(peakbot_MRM.Config.BATCHSIZE)],
                                "ref.rt"            : np.zeros((peakbot_MRM.Config.BATCHSIZE), dtype=float),
                                "ref.PeakForm"      : ["" for i in range(peakbot_MRM.Config.BATCHSIZE)], 
                                "ref.Rt shifts"     : ["" for i in range(peakbot_MRM.Config.BATCHSIZE)],
                                "ref.Note"          : ["" for i in range(peakbot_MRM.Config.BATCHSIZE)],
                                "loss.IOU_Area"     : np.ones((peakbot_MRM.Config.BATCHSIZE), dtype=float),}
                        curI = 0

                    assert curI < temp["channel.rt"].shape[0]
                    peakType = 0 if integrations[substance][sample]["foundPeak"] else 1

                    ## analytical data
                    temp["channel.rt"       ][curI,:] = rtsS
                    temp["channel.int"      ][curI,:] = eicS

                    ## integration data
                    temp["inte.peak"        ][curI, peakType] = 1
                    temp["inte.rtStart"     ][curI]   = bestRTStart
                    temp["inte.rtEnd"       ][curI]   = bestRTEnd
                    temp["inte.rtInds"      ][curI,0] = bestRTStartInd
                    temp["inte.rtInds"      ][curI,1] = bestRTEndInd                        
                    temp["inte.area"        ][curI]   = integrations[substance][sample]["area"]

                    ## reference data
                    temp["ref.substanceName"][curI] = substance
                    temp["ref.sample"       ][curI] = sample
                    temp["ref.rt"           ][curI] = substances[substance]["RT"]
                    temp["ref.PeakForm"     ][curI] = substances[substance]["PeakForm"] 
                    temp["ref.Rt shifts"    ][curI] = substances[substance]["Rt shifts"]
                    temp["ref.Note"         ][curI] = substances[substance]["Note"]

                    curI = curI + 1
                else:
                    np.set_printoptions(edgeitems=peakbot_MRM.Config.RTSLICES + 2, 
                        formatter=dict(float=lambda x: "%.3g" % x))
                    print(eicS)

            ## if batch has been filled, export it to a temporary file
            if curI >= peakbot_MRM.Config.BATCHSIZE:
                with open(os.path.join(instanceDir, "%s%d.pickle"%(peakbot_MRM.Config.INSTANCEPREFIX, cur)), "wb") as fout:
                    pickle.dump(temp, fout)
                    temp = None
                    curI = 0
                    cur += 1
    print("  | .. Exported %d batches each with %d instances"%(cur, peakbot_MRM.Config.RTSLICES))

    ## randomize instances
    print(".. shuffling instances\r", end="")
    peakbot_MRM.train.shuffleResultsSampleNames(instanceDir)
    peakbot_MRM.train.shuffleResults(instanceDir, steps=1E4, samplesToExchange=12)
    print("  | .. Instances shuffled")
    print("\n")



    ## train new model
    with tempfile.TemporaryDirectory() as tempTrainDir, tempfile.TemporaryDirectory() as tempValDir:
        peakbot_MRM.train.splitDSinto(instanceDir, 
                                        newDS1Path = tempTrainDir, newDS2Path = tempValDir, 
                                        copy = True, ratioDS1 = 0.7, verbose = False)
        nTrainBatches = len([f for f in os.listdir(tempTrainDir) if os.path.isfile(os.path.join(tempTrainDir, f))])
        nValBatches = len([f for f in os.listdir(tempValDir) if os.path.isfile(os.path.join(tempValDir, f))])           
        print("  | .. Randomly split dataset '%s' into a training and validation dataset with 0.7 and 0.3 parts of the instances "%expDir)
        print("  | .. There are %d training and %d validation batches available"%(nTrainBatches, nValBatches))
        
        addValDS = []
        addValDS.append({"folder": tempTrainDir, "name": "train", "numBatches": nTrainBatches-1})            
        addValDS.append({"folder": tempValDir, "name": "val", "numBatches": nValBatches-1})

        pb, hist = peakbot_MRM.trainPeakBotModel(trainInstancesPath = tempTrainDir,
                                                    addValidationInstances = addValDS,
                                                    logBaseDir = logDir,
                                                    everyNthEpoch = -1, 
                                                    verbose = True)

        pb.saveModelToFile(modelFile)
        print("Newly trained peakbot saved to file '%s'"%(modelFile))

        if histAll is None:
            histAll = hist
        else:
            histAll = histAll.append(hist, ignore_index=True)

        print("")
        print("")

        ### Summarize the training and validation metrices and losses
        histAll.to_pickle(historyObject)        

        ### Summarize and illustrate the results of the different training and validation dataset
        df = histAll
        df['ID'] = df.model.str.split('_').str[-1]
        df = df[df["metric"]!="loss"]
        df = df[df["set"]!="eV"]
        plot = (p9.ggplot(df, p9.aes("set", "value", colour="set"))
                + p9.geom_violin()
                + p9.geom_jitter()
                + p9.facet_wrap("~metric", scales="free_y", ncol=2)
                + p9.scale_x_discrete(limits=["train", "val"])
                + p9.ggtitle("Training losses/metrics") + p9.xlab("Training/Validation dataset") + p9.ylab("Value")
                + p9.theme(legend_position = "none", panel_spacing_x=0.5))
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "fig_SummaryPlot.png"), width=5.2, height=5, dpi=300)
    print("\n\n\n\n\n")

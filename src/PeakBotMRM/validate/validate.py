## General imports
from re import I
import tqdm
import os
import pickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
random.seed(2021)
import os
import math


import PeakBotMRM
from PeakBotMRM.core import tic, toc, TabLog, extractStandardizedEIC, getInteRTIndsOnStandardizedEIC
print("\n")



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

    substances               = PeakBotMRM.loadTargets(targetFile, excludeSubstances = excludeSubstances, includeSubstances = includeSubstances)
    substances, integrations = PeakBotMRM.loadIntegrations(substances, curatedPeaks)
    substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, samplesPath, expDir,
                                                             allowedMZOffset = allowedMZOffset,
                                                             MRMHeader = MRMHeader)


    print("Evaluating model (using predtrained model from '%s')"%(modelFile))
    offsetEIC = 0.2
    offsetRT1 = 0
    offsetRT2 = 0
    offsetRTMod = 1
    pbModelPred = PeakBotMRM.loadModel(modelFile, mode="predict", verbose = False)
    pbModelEval = PeakBotMRM.loadModel(modelFile, mode="training", verbose = False)
    allSubstances = set()
    allSamples = set()
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
        agreement = np.zeros((4))
        
        for samplei, sample in enumerate(integrations[substance].keys()):
            inte = integrations[substance][sample]
            allSamples.add(sample)
            assert len(inte["chrom"]) <= 1
            #integrations = {'foundPeak': False, 'rtstart': -1, 'rtend': -1, 'area': -1, 'chrom': [['- SRM SIC Q1=458.0 Q3=276.996 start=14.608683333 end=16.79575 (negative mode, collision-induced dissociation with 20.0 energy)', 458.0, 276.996, 14.608683333, 16.79575, 'negative', 20.0, 'collision-induced dissociation', '- SRM SIC Q1=458.0 Q3=276.996 start=14.608683333 end=16.79575', [(14.608684, 49.81321), (14.617084, 49.768093), (14.6255, 49.747154), (14.6339, 49.74894), (14.642317, 49.75564), (14.650717, 49.78736), (14.659133, 49.76708), (14.667533, 49.730103), (14.67595, 49.68629), (14.68435, 49.632336), (14.692767, 49.65427), (14.701167, 49.68037), (14.709583, 49.6406), (14.717983, 49.686207), (14.7264, 49.71693), (14.7348, 49.68437), (14.7432165, 49.748863), (14.751734, 49.73176), (14.760134, 49.740875), (14.768534, 49.768368), (14.776934, 49.72773), (14.78535, 49.731018), (14.79375, 49.733482), (14.80215, 49.70713), (14.81055, 49.68193), (14.81895, 49.694733), (14.82735, 49.694466), (14.83575, 49.76148), (14.84415, 49.77017), (14.85255, 49.764534), (14.86095, 49.748516), (14.86935, 49.692318), (14.87775, 49.789204), (14.88615, 49.76876), (14.894567, 49.78683), (14.903033, 49.80647), (14.911433, 49.746506), (14.919833, 49.78984), (14.928233, 49.788464), (14.93655, 49.808117), (14.94495, 49.751637), (14.95335, 49.721783), (14.96175, 49.725967), (14.97015, 49.650494), (14.97855, 49.65501), (14.98695, 49.60406), (14.99535, 49.670334), (15.0039835, 49.743942), (15.012383, 49.750694), (15.020783, 49.740856), (15.029183, 49.661087), (15.037583, 49.63553), (15.045983, 49.584503), (15.0545, 49.626423), (15.0629, 49.61743), (15.0713, 49.62126), (15.0797, 49.6702), (15.0881, 49.698406), (15.096517, 49.721992), (15.104917, 49.73892), (15.113317, 49.77488), (15.1217165, 49.7286), (15.130116, 49.704716), (15.138516, 49.672554), (15.146916, 49.63798), (15.155316, 49.647957), (15.1637335, 49.681396), (15.172367, 49.716274), (15.180767, 49.771694), (15.189167, 49.781685), (15.197567, 49.791546), (15.206083, 49.812492), (15.214483, 49.77248), (15.222867, 49.73366), (15.231267, 49.6815), (15.240033, 49.705647), (15.248433, 49.70874), (15.25685, 49.704823), (15.26525, 50.11055), (15.27365, 50.237015), (15.282066, 50.422485), (15.290717, 50.514126), (15.299117, 50.005386), (15.307533, 49.980366), (15.315933, 49.93789), (15.324333, 49.924706), (15.33275, 49.910423), (15.34115, 49.849247), (15.34955, 49.783913), (15.358183, 49.81969), (15.366583, 49.82353), (15.375, 49.841187), (15.383417, 49.868576), (15.391817, 49.828373), (15.400233, 49.80578), (15.408633, 49.758724), (15.41705, 49.754787), (15.425467, 49.759644), (15.4338665, 49.776245), (15.442284, 49.79674), (15.450684, 49.78991), (15.4591, 49.820778), (15.467517, 49.80407), (15.476, 49.82086), (15.4844, 49.9046), (15.4928, 49.874622), (15.5012, 49.898895), (15.509717, 49.905758), (15.518133, 49.8185), (15.526533, 49.76978), (15.53495, 49.735996), (15.54335, 49.755867), (15.551766, 49.78891), (15.560433, 49.795765), (15.568833, 49.809097), (15.57725, 49.790646), (15.58565, 49.78676), (15.59405, 49.800655), (15.602467, 49.827187), (15.610867, 49.832977), (15.619284, 49.7986), (15.627684, 49.77081), (15.6361, 49.711872), (15.6445, 49.707985), (15.652917, 49.683483), (15.661317, 49.712337), (15.669833, 49.7055), (15.67825, 49.666172), (15.68665, 49.72044), (15.69505, 49.726753), (15.703767, 49.741554), (15.712167, 49.765553), (15.720567, 49.788292), (15.728967, 49.809357), (15.73755, 49.784523), (15.745934, 49.804234), (15.7543335, 49.795048), (15.762733, 49.80904), (15.77125, 49.793976), (15.77965, 49.782684), (15.78805, 49.72643), (15.79645, 49.65997), (15.805083, 49.641506), (15.813467, 49.595314), (15.821867, 49.6229), (15.830267, 49.64364), (15.838834, 49.678123), (15.847217, 49.70794), (15.8556, 49.737103), (15.864, 49.765358), (15.872383, 49.765484), (15.880767, 49.7767), (15.889167, 49.75026), (15.89755, 49.671543), (15.906016, 49.63923), (15.9144, 49.58877), (15.9228, 49.620068), (15.931183, 49.68108), (15.939584, 49.712147), (15.947967, 49.774357), (15.956367, 49.729282), (15.96475, 49.738457), (15.97315, 49.730743), (15.981533, 49.687725), (15.990117, 49.710663), (15.998517, 49.692436), (16.006916, 49.704903), (16.0153, 49.708416), (16.023617, 49.709568), (16.032, 49.76107), (16.040382, 49.753117), (16.048767, 49.7517), (16.057283, 49.758125), (16.065683, 49.710365), (16.074066, 49.70181), (16.08245, 49.702435), (16.090834, 49.665512), (16.099234, 49.67054), (16.107616, 49.674706), (16.116, 49.68723), (16.1246, 49.715584), (16.132984, 49.766083), (16.141367, 49.780624), (16.14975, 49.798225), (16.158133, 49.80123), (16.166517, 49.784824), (16.175117, 49.787613), (16.183517, 49.79603), (16.1919, 49.77259), (16.200283, 49.756012), (16.208883, 49.736168), (16.217266, 49.699955), (16.22565, 49.699326), (16.234034, 49.723045), (16.242416, 49.746872), (16.2508, 49.785133), (16.259266, 49.80092), (16.26765, 49.75147), (16.276033, 49.710804), (16.284416, 49.701077), (16.2928, 49.71398), (16.301184, 49.716805), (16.309566, 49.728786), (16.31795, 49.699318), (16.326334, 49.67384), (16.334717, 49.664654), (16.3431, 49.662193), (16.351482, 49.688183), (16.359867, 49.691986), (16.36825, 49.684216), (16.376633, 49.639484), (16.385017, 49.59992), (16.3934, 49.58715), (16.401783, 49.600986), (16.410334, 49.618454), (16.4187, 49.62683), (16.427084, 49.629208), (16.435467, 49.637146), (16.44385, 49.63896), (16.452234, 49.624153), (16.460867, 49.58474), (16.46925, 49.56671), (16.477617, 49.57183), (16.485983, 49.57418), (16.494368, 49.591324), (16.502733, 49.592804), (16.511116, 49.601498), (16.519484, 49.60193), (16.52785, 49.605507), (16.536234, 49.579582), (16.5446, 49.569912), (16.552984, 49.58222), (16.56135, 49.576897), (16.569717, 49.620243), (16.5781, 49.624916), (16.586466, 49.62071), (16.59485, 49.626057), (16.603216, 49.620705), (16.61145, 49.623398), (16.619816, 49.645977), (16.6282, 49.627815), (16.636566, 49.603092), (16.64495, 49.57507), (16.653334, 49.522766), (16.6617, 49.51513), (16.670084, 49.513954), (16.678467, 49.54023), (16.686832, 49.573868), (16.695217, 49.58822), (16.7036, 49.59273), (16.711967, 49.594666), (16.72035, 49.602192), (16.728716, 49.589012), (16.7371, 49.620167), (16.745483, 49.587795), (16.75385, 49.589027), (16.762234, 49.570614), (16.770617, 49.280003), (16.778984, 49.620003), (16.787367, 49.600002), (16.79575, 49.640003)]]]}
            
            if len(inte["chrom"]) == 1:
                rts = inte["chrom"][0][9]["rts"]
                eic = inte["chrom"][0][9]["eic"]
                refRT = substances[substance]["RT"]
                
                ## standardize EIC
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                
                ## get integration results on standardized area
                bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                    getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    inte["foundPeak"], 
                                                    inte["rtstart"], 
                                                    inte["rtend"])
                
                temp["channel.int"][samplei, :] = eicS
                temp["inte.peak"][samplei,0] = 1 if peakType else 0
                temp["inte.peak"][samplei,1] = 1 if not peakType else 0
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
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT, scaleToOne = False, removeConstantOffset = False)
            
                ## get integration results on standardized area
                bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                    getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                    inte["foundPeak"], 
                                                    inte["rtstart"], 
                                                    inte["rtend"])
                    
                ## test if eic has detected signals                
                ppeakType = np.argmax(ppeakTypes[samplei,:])
                ppeakType = ppeakType == 0
                prtStartInd = round(prtStartInds[samplei])
                prtEndInd = round(prtEndInds[samplei])
                prtStart = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, prtStartInd))]
                prtEnd = rtsS[min(PeakBotMRM.Config.RTSLICES-1, max(0, prtEndInd))]
                agreement[0] = agreement[0] + (1 if peakType and ppeakType else 0)
                agreement[1] = agreement[1] + (1 if peakType and not ppeakType else 0)
                agreement[2] = agreement[2] + (1 if not peakType and ppeakType else 0)
                agreement[3] = agreement[3] + (1 if not peakType and not ppeakType else 0)

                inte["pred.rtstart"] = prtStart
                inte["pred.rtend"] = prtEnd
                inte["pred.foundPeak"] = ppeakType
                if inte["pred.foundPeak"]:
                    inte["pred.area"] = PeakBotMRM.integrateArea(eic, rts, prtStart, prtEnd, method = "linear")
                else:
                    inte["pred.area"] = -1
                if bestRTStart > 0:
                    inte["area."] = PeakBotMRM.integrateArea(eic, rts, bestRTStart, bestRTEnd, method = "linear")
                else:
                    inte["area."] = -1
                
                if plotSubstance == "all" or substance in plotSubstance:
                    ## plot results; find correct axis to plot to 
                    ax = ax1
                    axR = ax5
                    axS = ax9
                    if peakType:
                        ax = ax1 if ppeakType else ax2
                        axR = ax5 if ppeakType else ax6
                        axS = ax9 if ppeakType else ax10
                    else:
                        ax = ax3 if ppeakType else ax4
                        axR = ax7 if ppeakType else ax8
                        axS = ax11 if ppeakType else ax12
                                        
                    ax.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [offsetEIC*samplei, offsetEIC*samplei], "slategrey", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    axR.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [0,0], "slategrey", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    axS.plot([min(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod), max(t for t in rtsS if t > 0)+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod)], [0,0], "slategrey", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    
                    ## plot raw, scaled data according to classification prediction and integration result
                    b = min(eic)
                    m = max([i-b for i in eic])
                    ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], [(e-b)/m+offsetEIC*samplei for e in eic], "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts], [(e-b)/m+offsetEIC*samplei for e in eic], offsetEIC*samplei, facecolor='w', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                    ## add detected peak
                    if ppeakType:
                        ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if prtStart <= t <= prtEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                        ax.fill_between([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if prtStart <= t <= prtEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], offsetEIC*samplei, facecolor='yellowgreen', lw=0, zorder=(len(integrations[substance].keys())-samplei+1)*2-1)
                    ## add integration results
                    if peakType:            
                        ax.plot([t+offsetRT1*math.floor(samplei/offsetRTMod)+offsetRT2*(samplei%offsetRTMod) for t in rts if bestRTStart <= t <= bestRTEnd], [(e-b)/m+offsetEIC*samplei for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                                                
                    ## plot raw data
                    axR.plot(rts, eic, "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    ## add detected peak
                    if ppeakType:
                        axR.plot([t for t in rts if prtStart <= t <= prtEnd], [e for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    ## add integration results
                    if peakType:            
                        axR.plot([t for t in rts if bestRTStart <= t <= bestRTEnd], [e for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)

                    ## plot scaled data
                    ## add detected peak
                    minInt = min([e for e in eicS if e > 0])
                    maxInt = max([e-minInt for e in eicS])
                    axS.plot(rts, [(e-minInt)/maxInt for e in eic], "lightgrey", linewidth=.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    if ppeakType:
                        axS.plot([t for t in rts if prtStart <= t <= prtEnd], [(e-minInt)/maxInt for i, e in enumerate(eic) if prtStart <= rts[i] <= prtEnd], "olivedrab", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)
                    if peakType:
                        axS.plot([t for t in rts if bestRTStart <= t <= bestRTEnd], [(e-minInt)/maxInt for i, e in enumerate(eic) if bestRTStart <= rts[i] <= bestRTEnd], "k", linewidth=0.25, zorder=(len(integrations[substance].keys())-samplei+1)*2)

        
        if substance not in metricsTable.keys():
            metricsTable[substance] = {}
        metricsTable[substance] = {"CCA"        : metrics["pred.peak_categorical_accuracy"], 
                                   "MCC"        : metrics["pred.peak_MatthewsCorrelationCoefficient"],
                                   "RtMSE"      : metrics["pred.rtInds_MSE"], 
                                   "EICIOUPeaks": metrics["pred_EICIOUPeaks"],
                                   "EICIOU"     : metrics["pred_EICIOU"]}
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
            ax1.set_title('Integration peak\nPrediciton peak', loc="left")
            ax2.set_title('Integration peak\nPrediction no peak', loc="left")
            ax3.set_title('Integration no peak\nPrediction peak', loc="left")
            ax4.set_title('Integration no peak\nPrediction no peak', loc="left")
            fig.suptitle('%s\n%s\n%s\nGreen EIC and area: prediction; black EIC: manual integration; grey EIC: standardized EIC; light grey EIC: raw data'%(substance, substances[substance]["PeakForm"] + "; " + substances[substance]["Rt shifts"] + "; " + substances[substance]["Note"], "; ".join("%s: %.4g"%(k, v) for k, v in metricsTable[substance].items())), fontsize=14)

            plt.tight_layout()
            fig.savefig(os.path.join(expDir, "SubstanceFigures","%s.png"%substance), dpi = 600)
            print("  | .. plotted substance '%s'"%(substance))
            plt.close(fig)
    print("\n")

    allSubstances = list(allSubstances)
    allSamples = list(allSamples)
    with open(os.path.join(expDir, "results_IntePred.tsv"), "w") as fout:
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

    with open (os.path.join(expDir, "metricsTable.pickle"), "wb") as fout:
        pickle.dump(metricsTable, fout)
    with open (os.path.join(expDir, "metricsTable.pickle"), "rb") as fin:
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
            
        plt.colorbar(im)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len([rows[i]])))
        ax.set_xticklabels(cols)
        ax.set_yticklabels([rows[i]])
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        plt.tight_layout()
        fig.savefig(os.path.join(expDir,"Metric_%s.png"%(metric)), dpi = 300)
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
    fig.savefig(os.path.join(expDir,"Metrics.png"), dpi = 300)
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
    fig.savefig(os.path.join(expDir,"Metrics_short.png"), dpi = 300)
    plt.close()


    TabLog().exportToFile(os.path.join(expDir, "results.tsv"))
    print("All calculations took %.1f seconds"%(toc("Overall process")))





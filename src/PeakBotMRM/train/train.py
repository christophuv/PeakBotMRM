from PeakBotMRM.core import tic, toc, arg_find_nearest
import PeakBotMRM
from PeakBotMRM.core import extractStandardizedEIC, getInteRTIndsOnStandardizedEIC

import os
import plotnine as p9
import pandas as pd
#from umap import UMAP
#import pacmap
#from annoy import AnnoyIndex
import numpy as np
import tqdm
import random
random.seed(2021)

import math
import shutil
import pathlib
import portalocker





def compileInstanceDataset(substances, integrations, experimentName, dataset = None, 
                           addRandomNoise=False, maxRandFactor=0.1, maxNoiseLevelAdd=0.1, 
                           shiftRTs=False, maxShift=0.1, useEachInstanceNTimes=1, balanceReps = False, 
                           verbose = True, logPrefix = ""):
    template = None    
    curInstanceInd = 0
    totalInstances = 0
    if dataset is None:
        dataset = PeakBotMRM.MemoryDataset()
        if dataset.data != None:
            totalInstances = dataset.data["channel.rt"].shape[0]
    elif verbose:
        print(logPrefix, "  | .. %d instances already present in the dataset. Appending..."%(dataset.data["channel.rt"].shape[0]))
    
    
    if addRandomNoise and verbose:
        print(logPrefix, "  | .. Random noise will be added. The range of the randomly generated factors is %.3f - %.3f and the maximum randomly-generated noise added on top of the EICs is %.3f"%(1/(1 + maxRandFactor), 1 + maxRandFactor, maxNoiseLevelAdd))
    if shiftRTs:
        print(logPrefix, "  | .. Random RT shifts will be added. The range is -%.3f - %.3f minutes"%(maxShift, maxShift))
        print(logPrefix, "  | .. Chromatographic peaks with a shifted peak apex will first be corrected to the designated RT and then randomly moved for the training instance")
    if verbose: 
        print(logPrefix, "  | .. Each instance shall be used %d times and the peak/background classes shall%s be balanced"%(useEachInstanceNTimes, "" if balanceReps else " not"))
    
    useEachPeakInstanceNTimes = useEachInstanceNTimes
    useEachBackgroundInstanceNTimes = useEachInstanceNTimes
    if balanceReps:
        peaks = 0
        noPeaks = 0
        for substance in integrations.keys():
            for sample in integrations[substance].keys():
                inte = integrations[substance][sample]
                if len(inte["chrom"]) == 1:
                    if inte["foundPeak"]:
                        peaks += 1
                    else:
                        noPeaks += 1
        useEachPeakInstanceNTimes = int(round(useEachInstanceNTimes / (peaks / max(peaks, noPeaks))))
        useEachBackgroundInstanceNTimes = int(round(useEachInstanceNTimes / (noPeaks / max(peaks, noPeaks))))
    if verbose:
        print(logPrefix, "  | .. Each peak instance will be used %d times and each background instance %d times"%(useEachPeakInstanceNTimes, useEachBackgroundInstanceNTimes))
    for substance in tqdm.tqdm(integrations.keys(), desc=logPrefix + "   | .. augmenting"):
        for sample in integrations[substance].keys():
            inte = integrations[substance][sample]
            if len(inte["chrom"]) == 1:
                rts = inte["chrom"][0][9]["rts"]
                eic = inte["chrom"][0][9]["eic"]
                refRT = substances[substance]["RT"]
                
                ## generate replicates
                reps = useEachPeakInstanceNTimes if inte["foundPeak"] else useEachBackgroundInstanceNTimes
                for repi in range(reps):
                    ## add uniform Rt shift to EICs
                    artificialRTShift = 0
                    if repi > 0 and shiftRTs:
                        if inte["foundPeak"]:
                             ## shift according to peak boundaries
                            widthConstraint = 0.8 ## use entire chrom. peak width (=1) or less (0..1)
                            width = (inte["rtend"] - inte["rtstart"]) * widthConstraint
                            startRT = inte["rtstart"] + (1 - widthConstraint) / 2. * (inte["rtend"] - inte["rtstart"])
                            artificialRTShift = startRT + width * np.random.rand(1) - refRT
                        else:
                            artificialRTShift = np.random.rand(1) * 2 * maxShift - maxShift
                    
                    ## standardize EIC
                    rtsS, eicS = extractStandardizedEIC(eic, rts, refRT + artificialRTShift)
                    
                    ## get integration results on standardized area
                    bestRTInd, peakType, bestRTStartInd, bestRTEndInd, bestRTStart, bestRTEnd = \
                        getInteRTIndsOnStandardizedEIC(rtsS, eicS, refRT, 
                                                       inte["foundPeak"], 
                                                       inte["rtstart"], 
                                                       inte["rtend"])
                    
                    ## add random noise
                    if repi > 0 and addRandomNoise:
                        ## randomize signal intensitiers
                        if np.random.rand(1)[0] > 0.5:
                            eicS = eicS * (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                        else:
                            eicS = eicS / (1 + np.random.rand(eicS.shape[0]) * maxRandFactor)
                        
                        ## add noise on top of EIC
                        eicS = eicS + np.random.rand(eicS.shape[0]) * np.max(eicS) * maxNoiseLevelAdd
                    
                    ## test if eic has detected signals
                    if np.sum(eicS) > 0 and np.all(eicS >= 0):
                        ## add instance to training data
                        if template is None or curInstanceInd >= template["channel.rt"].shape[0]:
                            template = PeakBotMRM.getDatasetTemplate()
                            curInstanceInd = 0

                        ## analytical raw data
                        template["channel.rt"       ][curInstanceInd,:] = rtsS
                        template["channel.int"      ][curInstanceInd,:] = eicS

                        ## manual integration data
                        template["inte.peak"        ][curInstanceInd, 0 if inte["foundPeak"] else 1] = 1
                        template["inte.rtStart"     ][curInstanceInd]   = bestRTStart
                        template["inte.rtEnd"       ][curInstanceInd]   = bestRTEnd
                        template["inte.rtInds"      ][curInstanceInd,0] = bestRTStartInd
                        template["inte.rtInds"      ][curInstanceInd,1] = bestRTEndInd                        
                        template["inte.area"        ][curInstanceInd]   = inte["area"]

                        if PeakBotMRM.Config.INCLUDEMETAINFORMATION:
                            ## substance data
                            template["ref.substance" ][curInstanceInd] = substance
                            template["ref.sample"    ][curInstanceInd] = sample
                            template["ref.experiment"][curInstanceInd] = experimentName + ";" + sample + ";" + substance
                            template["ref.rt"        ][curInstanceInd] = substances[substance]["RT"]
                            template["ref.PeakForm"  ][curInstanceInd] = substances[substance]["PeakForm"] 
                            template["ref.Rt shifts" ][curInstanceInd] = substances[substance]["Rt shifts"]
                            template["ref.Note"      ][curInstanceInd] = substances[substance]["Note"]
                            template["loss.IOU_Area" ][curInstanceInd] = 1
                        
                        curInstanceInd += 1
                        totalInstances += 1
                    else:
                        np.set_printoptions(edgeitems=PeakBotMRM.Config.RTSLICES + 2, 
                            formatter=dict(float=lambda x: "%.3g" % x))
                        print(eicS)

                    ## if batch has been filled, export it to a temporary file
                    if curInstanceInd >= template["channel.rt"].shape[0]:
                        dataset.addData(template)
                        template = None
                        curInstanceInd = 0
    dataset.addData(template)
    dataset.removeOtherThan(0, totalInstances)
    if verbose:
        print(logPrefix, "  | .. Exported %d instances."%(dataset.getElements()))
        
    return dataset

def generateAndExportAugmentedInstancesForTraining(substances, integrations, experimentName, 
                                                   addRandomNoise, maxRandFactor, maxNoiseLevelAdd, 
                                                   shiftRTs, maxShift, useEachInstanceNTimes, balanceAugmentations, dataset = None, verbose = True, logPrefix = ""):
    if verbose: 
        print(logPrefix, "Exporting augmented instances for training")
    tic()
    dataset = compileInstanceDataset(substances, integrations, experimentName, dataset = dataset, addRandomNoise = addRandomNoise, maxRandFactor = maxRandFactor, maxNoiseLevelAdd = maxNoiseLevelAdd, shiftRTs = shiftRTs, maxShift = maxShift, useEachInstanceNTimes = useEachInstanceNTimes, balanceReps = balanceAugmentations, verbose = verbose, logPrefix = logPrefix)
    if verbose: 
        print(logPrefix, "  | .. took %.1f seconds"%(toc()))
        print(logPrefix)
    return dataset

def exportOriginalInstancesForValidation(substances, integrations, experimentName, dataset = None, verbose = True, logPrefix = ""):
    if verbose:
        print(logPrefix, "Exporting original instances for validation")
    tic()
    dataset = compileInstanceDataset(substances, integrations, experimentName, dataset = dataset, addRandomNoise = False, shiftRTs = False, verbose = verbose, logPrefix = logPrefix)
    if verbose: 
        print(logPrefix, "  | .. took %.1f seconds"%(toc()))
        print(logPrefix)
    return dataset

def constrainAndBalanceDataset(balanceDataset, checkPeakAttributes, substances, integrations, verbose = True, logPrefix = ""):
    if verbose: 
        print(logPrefix, "Balancing training dataset (and applying optional peak statistic filter criteria)")
    tic()
    peaks = []
    noPeaks = []
    notUsed = 0
    for substance in tqdm.tqdm(substances.values(), desc=logPrefix + "   | .. balancing"):
        subName = substance["Name"]
        if subName in integrations.keys():
            for sample in integrations[subName].keys():
                if integrations[subName][sample]["foundPeak"]:
                    inte = integrations[subName][sample]
                    if len(inte["chrom"]) == 1:
                        rts = inte["chrom"][0][9]["rts"]
                        eic = inte["chrom"][0][9]["eic"]
                        refRT = substances[subName]["RT"]
                        rtstart = inte["rtstart"]
                        rtend = inte["rtend"]
                        
                        rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                        eicS[rtsS < inte["rtstart"]] = 0
                        eicS[rtsS > inte["rtend"]] = 0
                        apexRT = rtsS[np.argmax(eicS)]
                        
                        intLeft  = eicS[arg_find_nearest(rtsS, rtstart)]
                        intRight = eicS[arg_find_nearest(rtsS, rtend)]
                        intApex  = eicS[arg_find_nearest(rtsS, apexRT)]
                        
                        peakWidth = rtend - rtstart
                        centerOffset = apexRT - refRT
                        peakLeftInflection = intLeft - apexRT
                        peakRightInflection = intRight - apexRT
                        leftIntensityRatio = intApex/intLeft if intLeft > 0 else np.Inf
                        rightIntensityRatio = intApex/intRight if intRight > 0 else np.Inf
                        
                        if checkPeakAttributes is None or checkPeakAttributes(peakWidth, centerOffset, peakLeftInflection, peakRightInflection, leftIntensityRatio, rightIntensityRatio, eicS, rtsS):
                            peaks.append((subName, sample))
                        else:
                            notUsed = notUsed + 1
                else:
                    noPeaks.append((subName, sample))
    if verbose: 
        print(logPrefix, "  | .. there are %d peaks and %d backgrounds in the dataset."%(len(peaks), len(noPeaks)))
    if checkPeakAttributes is not None and verbose:
        print(logPrefix, "  | .. .. %d peaks were not used due to peak abnormalities according to the user-provided peak-quality function checkPeakAttributes."%(notUsed))
    random.shuffle(peaks)
    random.shuffle(noPeaks)
    a = min(len(peaks), len(noPeaks))
    if balanceDataset:
        peaks = peaks[:a]
        noPeaks = noPeaks[:a]
    inte2 = {}
    for substance, sample in peaks:
        if substance not in inte2.keys():
            inte2[substance] = {}
        inte2[substance][sample] = integrations[substance][sample]        
    for substance, sample in noPeaks:
        if substance not in inte2.keys():
            inte2[substance] = {}
        inte2[substance][sample] = integrations[substance][sample]
    integrations = inte2
    peaks = []
    noPeaks = []
    for substance in substances.values():
        if substance["Name"] in integrations.keys():
            for sample in integrations[substance["Name"]].keys():
                if integrations[substance["Name"]][sample]["foundPeak"]:
                    peaks.append((substance["Name"], sample))
                else:
                    noPeaks.append((substance["Name"], sample))
    if verbose:
        if balanceDataset:
            print(logPrefix, "  | .. balanced dataset to %d peaks and %d backgrounds"%(len(peaks), len(noPeaks)))
        else:
            print(logPrefix, "  | .. dataset not balanced with %d peaks and %d backgrounds"%(len(peaks), len(noPeaks)))
        print(logPrefix, "  | .. took %.1f seconds"%(toc()))
        print(logPrefix)
    return integrations

def investigatePeakMetrics(expDir, substances, integrations, expName = "", verbose = True, logPrefix = ""):
    if verbose:
        print(logPrefix, "Peak statistics for '%s'"%(expName))
    tic()
    stats = {"hasPeak":0, "hasNoPeak":0, "peakProperties":[]}
    X = np.zeros((10000, PeakBotMRM.Config.RTSLICES), dtype=np.float)
    Y = np.zeros((10000), dtype=np.float)
    ysample = []
    ysubstance = []
    xcur = 0
    for substance in tqdm.tqdm(integrations.keys(), desc="  | .. calculating"):
        for sample in integrations[substance].keys():
            inte = integrations[substance][sample]
            if substance in substances.keys() and len(inte["chrom"]) == 1:
                rts = inte["chrom"][0][9]["rts"]
                eic = inte["chrom"][0][9]["eic"]
                refRT = substances[substance]["RT"]                
                rtsS, eicS = extractStandardizedEIC(eic, rts, refRT)
                
                temp = refRT 
                            
                if inte["foundPeak"]:
                    stats["hasPeak"] = stats["hasPeak"] + 1
                    
                    eicS[rtsS < inte["rtstart"]] = 0
                    eicS[rtsS > inte["rtend"]] = 0
                    apexRT = rtsS[np.argmax(eicS)]
                    
                    intLeft  = eicS[np.argmin(np.abs(rtsS - inte["rtstart"]))]
                    intRight = eicS[np.argmin(np.abs(rtsS - inte["rtend"]))]
                    intApex  = eicS[np.argmin(np.abs(rtsS - apexRT))]
                    
                    peakWidth = inte["rtend"] - inte["rtstart"]
                    peakWidthScans = np.argmin(np.abs(rtsS - inte["rtend"])) - np.argmin(np.abs(rtsS - inte["rtstart"]))
                    centerOffset = apexRT - refRT
                    peakLeftInflection = inte["rtstart"] - apexRT
                    peakRightInflection = inte["rtend"] - apexRT
                    leftIntensityRatio = intApex/intLeft if intLeft > 0 else np.Inf
                    rightIntensityRatio = intApex/intRight if intRight > 0 else np.Inf                    
                    
                    stats["peakProperties"].append([sample, "peakWidth", peakWidth])
                    stats["peakProperties"].append([sample, "peakWidthScans", peakWidthScans])
                    stats["peakProperties"].append([sample, "apexReferenceOffset", centerOffset])
                    stats["peakProperties"].append([sample, "peakLeftInflection", peakLeftInflection])
                    stats["peakProperties"].append([sample, "peakRightInflection", peakRightInflection])
                    stats["peakProperties"].append([sample, "peakBorderLeftIntensityRatio", leftIntensityRatio])
                    stats["peakProperties"].append([sample, "peakBorderRightIntensityRatio", rightIntensityRatio])
                    if intLeft > 0:
                        stats["peakProperties"].append([sample, "peakBorderLeftIntensityRatioNonInf", intApex/intLeft])
                    if intRight > 0:
                        stats["peakProperties"].append([sample, "peakBorderRightIntensityRatioNonInf", intApex/intRight])
                    stats["peakProperties"].append([sample, "eicStandStartToRef", min(rtsS[rtsS>0]) - refRT])
                    stats["peakProperties"].append([sample, "eicStandEndToRef", max(rtsS) - refRT])
                    
                    if False:
                        temp = apexRT
                    
                else:
                    stats["hasNoPeak"] = stats["hasNoPeak"] + 1
                
                if xcur >= X.shape[0]:
                    X = np.concatenate((X, np.zeros((10000, PeakBotMRM.Config.RTSLICES), dtype=np.float)), axis=0)
                    Y = np.concatenate((Y, np.zeros((10000), dtype=np.float)), axis=0)
                
                rtsS, eicS = extractStandardizedEIC(eic, rts, temp)
                X[xcur,:] = eicS
                X[xcur,:] = X[xcur,:] - np.min(X[xcur,:])
                X[xcur,:] = X[xcur,:] / np.max(X[xcur,:])
                Y[xcur] = 0 if inte["foundPeak"] else 1
                ysample.append("Cal" if "CAL" in sample else ("Nist" if "NIST" in sample else "sample"))
                ysubstance.append(substance)
                xcur = xcur + 1
    Xori = X[0:xcur,:]
    Yori = Y[0:xcur]
    
    
    ## PacMap embedding (deactivated due to problems with the library)
    if False:
        for s in set(["allSubstances"] + ysubstance):
            if verbose:
                print(logPrefix, "  | .. generating Pacmap embedding..")
            
            if s != "allSubstances":
                inds = [ind for ind, i in enumerate(ysubstance) if i == s]
                if sum(inds) > 0:
                    X = Xori[inds, :]
                    Y = Yori[inds]
            else:
                X = Xori
                Y = Yori
            
            ## adapted from https://pypi.org/project/pacmap/
            
            n, dim = X.shape
            n_neighbors = 10
            tree = AnnoyIndex(dim, metric='euclidean')
            for i in range(n):
                tree.add_item(i, X[i, :])
            tree.build(20)

            nbrs = np.zeros((n, 20), dtype=np.int32)
            for i in range(n):
                nbrs_ = tree.get_nns_by_item(i, 20 + 1) # The first nbr is always the point itself
                nbrs[i, :] = nbrs_[1:]

            scaled_dist = np.ones((n, n_neighbors)) # No scaling is needed

            # Type casting is needed for numba acceleration
            X = X.astype(np.float32)
            scaled_dist = scaled_dist.astype(np.float32)
            # make sure n_neighbors is the same number you want when fitting the data
            pair_neighbors = pacmap.sample_neighbors_pair(X, scaled_dist, nbrs, np.int32(n_neighbors))

            # initializing the pacmap instance
            # feed the pair_neighbors into the instance
            embedding = pacmap.PaCMAP(n_dims=2, n_neighbors=n_neighbors, MN_ratio=0.5, FP_ratio=2.0, pair_neighbors=pair_neighbors) 

            # fit the data (The index of transformed data corresponds to the index of the original data)
            X_trans = embedding.fit_transform(X, init="pca")
            print(X.shape, Y.shape, X_trans.shape)

            df=pd.DataFrame({"x": X_trans[:,0], "y": X_trans[:,1], "label": Y, "sample": ysample if s == "allSubstances" else [ysample[i] for i in inds], "substance": ysubstance if s == "allSubstances" else [ysubstance[i] for i in inds]})
            df['label'] = df['label'].astype(int)
            df.sort_values(by='label', axis=0, ascending=True, inplace=True)
            df.to_pickle(os.path.join(expDir, "pacmap.pd.pickle"))
            plot = (p9.ggplot(df, p9.aes("x", "y", color="label"))
                    + p9.geom_point(alpha = 0.1 if s == "allSubstances" else 1)
                    + p9.ggtitle("PaCMAP of EICs") + p9.xlab("PaCMAP 1") + p9.ylab("PaCMAP 2"))
            p9.options.figure_size = (5.2,5)
            p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_PaACMAP.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
            p9.options.figure_size = (5.2,5)
            p9.ggsave(plot=(plot + p9.facet_wrap("~label")), filename=os.path.join(expDir, "SubstanceFigures", "%s_PACMAP_%s.png"%(expName, s)), width=5.2, height=5, dpi=300, verbose=False)
            
            plot = (p9.ggplot(df, p9.aes("x", "y", color="sample"))
                    + p9.geom_point(alpha = 0.1 if s == "allSubstances" else 1)
                    + p9.facet_wrap("~label")
                    + p9.theme(legend_position="bottom")
                    + p9.ggtitle("PaCMAP of EICs (color: substance)") + p9.xlab("PaCMAP 1") + p9.ylab("PaCMAP 2"))
            p9.options.figure_size = (10,8)
            p9.ggsave(plot=(plot + p9.facet_wrap("~label")), filename=os.path.join(expDir, "SubstanceFigures", "%s_PACMAPfacLabSub_%s.png"%(expName, s)), width=10, height=8, dpi=300, verbose=False)
        
    ## UMAP embedding (deactivated due to problems with the library)
    if False:        
        if verbose: 
            print(logPrefix, "  | .. generating UMAP EIC embedding..")
        
        ## adapted from https://towardsdatascience.com/umap-dimensionality-reduction-an-incredibly-robust-machine-learning-algorithm-b5acb01de568
        
        reducer = UMAP(n_neighbors=20, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                n_components=2, # default 2, The dimension of the space to embed into.
                metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
                n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
                learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
                init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
                spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
                metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
                #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
                verbose=False, # default False, Controls verbosity of logging.
                unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
                )

        # Fit and transform the data
        X_trans = reducer.fit_transform(X)

        df=pd.DataFrame({"x": X_trans[:,0], "y": X_trans[:,1], "label": Y, "sample": ysample, "substance": ysubstance})
        df['label'] = df['label'].astype(int)
        df.sort_values(by='label', axis=0, ascending=True, inplace=True)
        df.to_pickle(os.path.join(expDir, "umap.pd.pickle"))
        plot = (p9.ggplot(df, p9.aes("x", "y", color="label"))
                + p9.geom_point(alpha = 0.1)
                + p9.ggtitle("UMAP of EICs") + p9.xlab("UMAP 1") + p9.ylab("UMAP 2"))
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_UMAP.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=(plot + p9.facet_wrap("~label")), filename=os.path.join(expDir, "%s_peakStats_UMAPfacLab.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
        

        # Fit and transform the data
        X_trans = reducer.fit_transform(X, Y)

        df=pd.DataFrame({"x": X_trans[:,0], "y": X_trans[:,1], "label": Y, "sample": ysample, "substance": ysubstance})
        df['label'] = df['label'].astype(int)
        df.sort_values(by='label', axis=0, ascending=True, inplace=True)
        df.to_pickle(os.path.join(expDir, "umapsup.pd.pickle"))
        plot = (p9.ggplot(df, p9.aes("x", "y", color="label"))
                + p9.geom_point(alpha = 0.1)
                + p9.ggtitle("UMAP of EICs") + p9.xlab("UMAP 1") + p9.ylab("UMAP 2"))
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_UMAPsup.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=(plot + p9.facet_wrap("~label")), filename=os.path.join(expDir, "%s_peakStats_UMAPsupfacLab.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
        
        
    tf = pd.DataFrame(stats["peakProperties"], columns = ["sample", "type", "value"])
    if verbose:
        print(logPrefix, "  | .. There are %d peaks and %d Nopeaks. An overview of the peak stats has been saved to '%s'"%(stats["hasPeak"], stats["hasNoPeak"], os.path.join(expDir, "%s_peakStat.png"%(expName))))
        print(logPrefix, "  | .. .. The distribution of the peaks' properties (offset of apex to expected rt, left and right extends relative to peak apex, peak widths) are: (in minutes)")
        print(logPrefix, tf.groupby("type").describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        
    df = tf.copy()
    df.drop(df[df.type == "peakWidthScans"].index, inplace=True)
    df.drop(df[df.type == "peakBorderLeftIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "peakBorderRightIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "peakBorderLeftIntensityRatioNonInf"].index, inplace=True)
    df.drop(df[df.type == "peakBorderRightIntensityRatioNonInf"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free_y", ncol=2)
            + p9.ggtitle("Peak metrics") + p9.xlab("retention time (minutes)") + p9.ylab("Frequency")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_1.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
    
    df = tf.copy()
    df.drop(df[df.type != "peakWidthScans"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free", ncol=2)
            + p9.ggtitle("Peak metrics") + p9.xlab("peak width (scans)") + p9.ylab("Frequency")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_2.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
    
    df = tf.copy()
    df.drop(df[df.type == "apexReferenceOffset"].index, inplace=True)
    df.drop(df[df.type == "eicStandEndToRef"].index, inplace=True)
    df.drop(df[df.type == "eicStandStartToRef"].index, inplace=True)
    df.drop(df[df.type == "peakLeftInflection"].index, inplace=True)
    df.drop(df[df.type == "peakRightInflection"].index, inplace=True)
    df.drop(df[df.type == "peakWidth"].index, inplace=True)
    df.drop(df[df.type == "peakWidthScans"].index, inplace=True)
    df.drop(df[df.type == "peakBorderLeftIntensityRatio"].index, inplace=True)
    df.drop(df[df.type == "peakBorderRightIntensityRatio"].index, inplace=True)
    plot = (p9.ggplot(df, p9.aes("value"))
            + p9.geom_histogram()
            + p9.facet_wrap("~type", scales="free", ncol=2)
            + p9.scales.scale_x_log10() + p9.scales.scale_y_log10()
            + p9.ggtitle("Peak metrics") + p9.xlab("-") + p9.ylab("Frequency")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(expDir, "%s_peakStats_3.png"%(expName)), width=5.2, height=5, dpi=300, verbose=False)
    
    if verbose:
        print(logPrefix, "  | .. plotted peak metrics to file '%s'"%(os.path.join(expDir, "%s_peakStat.png"%(expName))))
        print(logPrefix, "  | .. took %.1f seconds"%toc())
        print(logPrefix)

def plotHistory(histObjectFile, plotFile, verbose = True, logPrefix = ""):
    
    histAll = None
    with portalocker.Lock(histObjectFile, timeout = 60, check_interval = 2) as fh:
        histAll = pd.read_pickle(histObjectFile)
        
    
    ### Summarize and illustrate the results of the different training and validation dataset
    df = histAll
    df['ID'] = df.model.str.split('_').str[-1]
    df = df[df["metric"]!="loss"]
    plot = (p9.ggplot(df, p9.aes("set", "value", colour="set"))
            #+ p9.geom_violin()
            + p9.geom_jitter(height=0, alpha=0.5)
            + p9.facet_grid("metric~comment", scales="free_y")
            + p9.ggtitle("Training losses/metrics") + p9.xlab("Training/Validation dataset") + p9.ylab("Value")
            + p9.theme(legend_position = "none", axis_text_x=p9.element_text(angle=45)))
    p9.options.figure_size = (5.2, 7)
    p9.ggsave(plot=plot, filename="%s.png"%(plotFile), width=40, height=12, dpi=300, limitsize=False, verbose=False)
    
    df = df[[i in ["Sensitivity (peaks)", "Specificity (no peaks)"] for i in df.metric]]
    df.value = df.apply(lambda x: x.value if x.metric != "Specificity (no peaks)" else 1 - x.value, axis=1)
    df.metric = df.apply(lambda x: "FPR" if x.metric == "Specificity (no peaks)" else x.metric, axis=1)
    df.metric = df.apply(lambda x: "TPR" if x.metric == "Sensitivity (peaks)" else x.metric, axis=1)
    df = df.pivot(index=["model", "set", "comment"], columns="metric", values="value")
    df.reset_index(inplace=True)
    
    if False:
        for model in set(list(df.model)):
            for s in set(list(df.set)):
                df = df.append({"model": model, "set": s, "FPR": 0, "TPR": 0}, ignore_index=True)
                df = df.append({"model": model, "set": s, "FPR": 1, "TPR": 1}, ignore_index=True)
        df = df.append({"model": "", "set": "", "FPR": 0, "TPR": 0}, ignore_index=True)
        df = df.append({"model": "", "set": "", "FPR": 1, "TPR": 1}, ignore_index=True)
        df = df.sort_values(["FPR", "TPR"])
        
    plot = (p9.ggplot(df, p9.aes("FPR", "TPR", colour="comment", shape="set", group="model"))
            + p9.geom_point(alpha=0.5)
            + p9.geom_line(alpha=0.5)
            + p9.facet_wrap("~comment", ncol=4)
            + p9.ggtitle("ROC") + p9.xlab("FPR") + p9.ylab("TPR") 
            )
    p9.options.figure_size = (5.2, 7)
    p9.ggsave(plot=plot, filename="%s_ROC.png"%(plotFile), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
    p9.ggsave(plot=p, filename="%s_ROC_zoomed.png"%(plotFile), width=20, height=20, dpi=300, limitsize=False, verbose=False)
    
    for k in set(df["set"]):
        temp = df[df["set"] == k]    
        plot = (p9.ggplot(temp, p9.aes("FPR", "TPR", colour="comment", shape="set", group="model"))
                + p9.geom_point(alpha=0.5)
                + p9.geom_line(alpha=0.5)
                + p9.facet_wrap("~comment", ncol=4)
                + p9.ggtitle("ROC") + p9.xlab("FPR") + p9.ylab("TPR") 
                )
        p9.options.figure_size = (5.2, 7)
        p9.ggsave(plot=plot, filename="%s_ROC_%s.png"%(plotFile, k), width=20, height=20, dpi=300, limitsize=False, verbose=False)
        p = (plot + p9.scales.xlim(0,0.21) + p9.scales.ylim(0.9,1))
        p9.ggsave(plot=p, filename="%s_ROC_zoomed_%s.png"%(plotFile, k), width=20, height=20, dpi=300, limitsize=False, verbose=False)

def trainPeakBotMRMModel(expName, trainDSs, valDSs, modelFile, expDir = None, logDir = None, historyFile = None, 
                         MRMHeader = "- SRM SIC Q1=(\\d+[.]\\d+) Q3=(\\d+[.]\\d+) start=(\\d+[.]\\d+) end=(\\d+[.]\\d+)",
                         allowedMZOffset = 0.05, balanceDataset = False, balanceAugmentations = True,
                         addRandomNoise = True, maxRandFactor = 0.1, maxNoiseLevelAdd=0.1, shiftRTs = True, maxShift = 0.15, useEachInstanceNTimes = 5, 
                         checkPeakAttributes = None, showPeakMetrics = True, 
                         comment="None", useDSForTraining = "augmented", 
                         verbose = True, logPrefix = ""):
    tic("Overall process")
    
    if expDir is None:
        expDir = os.path.join(".", expName)
    if logDir is None:
        logDir = os.path.join(expDir, "log")
    if historyFile is None:
        historyFile = os.path.join(expDir, "History.pandas.pickle")
    
    try:
        if not os.path.isdir(expDir):
            os.mkdir(expDir)
    except:
        print("Could not generate experiment directory '%s'"%(expDir))
        raise
    try:
        if not os.path.isdir(os.path.join(expDir, "SubstanceFigures")):
            os.mkdir(os.path.join(expDir, "SubstanceFigures"))
    except:
        print("Could not generate substance figure directory '%s'"%(os.path.join(expDir, "SubstanceFigures")))   
        raise
        
        
    print("Training model from experiment")
    print("  | .. Parameters")
    print("  | .. .. expName: '%s'"%(expName))
    print("  | .. .. modelFile: '%s'"%(modelFile))
    print("  | .. .. expDir: '%s'"%(expDir))
    print("  | .. .. logDir: '%s'"%(logDir))
    print("  | .. .. MRMHeader: '%s'"%(MRMHeader))
    print("  | .. .. allowedMZOffset: '%s'"%(allowedMZOffset))
    print("  | .. .. addRandomNoise: '%s'"%(addRandomNoise))
    
    print("  | .. Check peak attributes")
    print("  | .. .. %s"%("not checking and not restricting" if checkPeakAttributes is None else "checking and restricting with user-provided function checkPeakAttributes"))
    
    if balanceDataset or balanceAugmentations:
        print("  | .. Balancing dataset")
        if balanceDataset:
            print("  | .. .. the instances will be balanced so that an equal number of peak and background instances are present before augmentation")
        if balanceAugmentations:
            print("  | .. .. the instances will be balanced during instance augmentation. peaks or backgrounds underrepresented will be used several times more than the other class")
    
    print("  | .. Augmenting")
    if addRandomNoise:
        print("  | .. .. adding random noise")
        print("  | .. .. maxRandFactor: '%s'"%(maxRandFactor))
        print("  | .. .. maximum noise add level (relative to most abundant signal) '%s'"%(maxNoiseLevelAdd))
    if shiftRTs:
        print("  | .. .. shifting RTs of background instances")
        print("  | .. .. maxShift: '%s'"%(maxShift))
    print("  | .. Using each instance %d times for training"%(useEachInstanceNTimes))
    print("\n")
    
    
    print("PeakBotMRM configuration")
    print(PeakBotMRM.Config.getAsStringFancy())
    print("\n")
    
    ## Generate training instances
    validationDSs = []
    trainDataset = PeakBotMRM.MemoryDataset()
    for trainDS in trainDSs:
            
        print("Adding training dataset '%s'"%(trainDS["DSName"]))
        
        substances               = PeakBotMRM.loadTargets(trainDS["targetFile"], 
                                                          excludeSubstances = trainDS["excludeSubstances"], 
                                                          includeSubstances = trainDS["includeSubstances"], logPrefix = "  | ..")
        substances, integrations = PeakBotMRM.loadIntegrations(substances, trainDS["curatedPeaks"], logPrefix = "  | ..")
        substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, trainDS["samplesPath"],
                                                                allowedMZOffset = allowedMZOffset, 
                                                                MRMHeader = MRMHeader, logPrefix = "  | ..")
        if showPeakMetrics:
            investigatePeakMetrics(expDir, substances, integrations, expName = "%s"%(trainDS["DSName"]), logPrefix = "  | ..")
        
        integrations = constrainAndBalanceDataset(balanceDataset, checkPeakAttributes, substances, integrations, logPrefix = "  | ..")
        
        dataset = exportOriginalInstancesForValidation(substances, integrations, "Train_Ori_%s"%(trainDS["DSName"]), logPrefix = "  | ..")
        dataset.shuffle()
        dataset.setName("%s_Train_Ori"%(trainDS["DSName"]))
        validationDSs.append(dataset)
        if useDSForTraining.lower() == "original":
            trainDataset.addData(dataset.data)
        
        dataset = generateAndExportAugmentedInstancesForTraining(substances, integrations, "Train_Aug_%s"%(trainDS["DSName"]), addRandomNoise, maxRandFactor, maxNoiseLevelAdd, 
                                                                 shiftRTs, maxShift, useEachInstanceNTimes, balanceAugmentations, logPrefix = "  | ..")
        dataset.shuffle()
        dataset.setName("%s_Train_Aug"%(trainDS["DSName"]))
        validationDSs.append(dataset)
        if useDSForTraining.lower() == "augmented":
            trainDataset.addData(dataset.data)
        
        print("")

    ## prepare/split training dataset into train and validation set
    print("Dataset for training")
    tic()
    if useDSForTraining.lower() == "original":
        print("  | .. The original, unmodified instances are used")
    elif useDSForTraining.lower() == "augmented":
        print("  | .. The augmented instances are used")
    print("  | .. Shuffling training instances")
    trainDataset.shuffle()  
    print("  | .. took %.1f seconds"%(toc()))
    tic()  
    splitRatio = 0.7
    trainDataset, valDataset = trainDataset.split(ratio = splitRatio)
    trainDataset.setName("Train_split_%s"%(expName))
    valDataset.setName("Train_split_%s"%(expName))
    validationDSs.append(trainDataset)
    validationDSs.append(valDataset)
    print("  | .. Randomly split dataset into a training and validation dataset with %.1f and %.1f parts of the instances "%(splitRatio, 1-splitRatio))
    print("  | .. There are %d training (%s) and %d validation (%s) batches available"%(trainDataset.data["channel.rt"].shape[0], trainDataset.name, valDataset.data["channel.rt"].shape[0], valDataset.name))
    print("  | .. took %.1f seconds"%(toc()))
    print("\n")
    
    
    ## add other datasets for validation metrics
    if valDSs is not None:
        for valDS in valDSs:
            
            print("Adding additional validation dataset '%s'"%(valDS["DSName"]))
            
            substances               = PeakBotMRM.loadTargets(valDS["targetFile"], 
                                                              excludeSubstances = valDS["excludeSubstances"], 
                                                              includeSubstances = valDS["includeSubstances"], logPrefix = "  | ..")
            substances, integrations = PeakBotMRM.loadIntegrations(substances, valDS["curatedPeaks"], logPrefix = "  | ..")
            substances, integrations = PeakBotMRM.loadChromatograms(substances, integrations, valDS["samplesPath"],
                                                                    allowedMZOffset = allowedMZOffset, 
                                                                    MRMHeader = MRMHeader, logPrefix = "  | ..")
            if showPeakMetrics:
                investigatePeakMetrics(expDir, substances, integrations, expName = "%s"%(valDS["DSName"]), logPrefix = "  | ..")
            
            integrations = constrainAndBalanceDataset(False, checkPeakAttributes, substances, integrations, logPrefix = "  | ..")
            
            dataset = exportOriginalInstancesForValidation(substances, integrations, "AddVal_Ori_%s"%(valDS["DSName"]), logPrefix = "  | ..")
            dataset.shuffle()
            dataset.setName("%s_AddVal_Ori"%(valDS["DSName"]))
            validationDSs.append(dataset)
    
            print("")
    
    print("Preparation for training took %.1f seconds"%(toc("Overall process")))
    print("")       
    
    ## Train new peakbotMRM model
    pb, chist, modelName = PeakBotMRM.trainPeakBotMRMModel(modelName = os.path.basename(modelFile), 
                                                           trainDataset = trainDataset,
                                                           addValidationDatasets = validationDSs,
                                                           logBaseDir = logDir,
                                                           everyNthEpoch = -1, 
                                                           verbose = True)

    pb.saveModelToFile(modelFile)
    print("Newly trained PeakBotMRM saved to file '%s'"%(modelFile))
    print("\n")

    ## add current history
    chist["comment"] = comment
    chist["modelName"] = modelName

    ### Summarize the training and validation metrices and losses
    if not os.path.exists(historyFile):
        chist.to_pickle(historyFile)
    else:
        with portalocker.Lock(historyFile, timeout = 60, check_interval = 2) as fh:
            if os.path.exists(historyFile):
                history = pd.read_pickle(historyFile)
                history = history.append(chist, ignore_index=True)
            else:
                history = chist
            history.to_pickle(historyFile)

    print("\n\n\n")
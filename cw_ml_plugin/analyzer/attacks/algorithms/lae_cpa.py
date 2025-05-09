import numpy as np
import math

from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase
from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.logging import * # type: ignore

from cw_ml_plugin.analyzer.preprocessing.poi_slice import Slice
from cw_ml_plugin.analyzer.preprocessing.convolve import Convolve
from cw_ml_plugin.analyzer.preprocessing.normalize import Normalize
from cw_ml_plugin.analyzer.preprocessing.hd_sort import HD_Sort
from cw_ml_plugin.analyzer.preprocessing.group_and_label import Group_and_Label
from cw_ml_plugin.analyzer.preprocessing.autoencorder import Auto_Encorder


class CPA_LAEOneSubkey(object):
    """This class is the basic progressive CPA attack, capable of adding traces onto a variable with previous data"""
    def __init__(self, model, project:Project, parameters):
        self.model = model
        self._project: Project = project
        self.parameters = parameters
        
        # Doesn't rely on nbkey or nbyte
        self.sum_trace = [0]
        self.square_sum_trace = [0]
        # Relies on nkey or nbyte
        self.square_sum_hyp = [0] * self.model.getPermPerSubkey()
        self.sum_hyp = [0] * self.model.getPermPerSubkey()
        self.sum_ht = [0] * self.model.getPermPerSubkey()
        
        self.totalTraces = 0
        self.modelstate = {'knownkey':None}
        
        
    def preprocess(self):
        slice = Slice(self._project)
        print(self.parameters._point_range)
        slice.poi = self.parameters._point_range
        slice.trace_num = self.parameters._trace_range
        slice_proj = slice.preprocess()
        
        if self.parameters._convolve_vector:
            convolve = Convolve(slice_proj)
            convolve.convolve_mode = 'same'
            convolve.weight_vector = self.parameters._convolve_vector
            slice_proj = convolve.preprocess()

        norm = Normalize(slice_proj)
        self.norm_proj = norm.preprocess()

    def oneSubkey(self, bnum, pointRange, traces_all, tend, tstart, plaintexts, ciphertexts, knownkeys, progressBar, state, pbcnt, tracerange):
        diffs = [0] * self.model.getPermPerSubkey()
        numtraces = tend - tstart
        self.totalTraces += numtraces

        if pointRange == None:
            traces = traces_all
        else:
            traces = traces_all[:, pointRange[0] : pointRange[1]]


        #Onesubkeyの名前的にもこのループは必要かな
        for key in range(0, self.model.getPermPerSubkey()):

            #Formula for CPA & description found in "Power Analysis Attacks"
            # by Mangard et al, page 124, formula 6.2.
            #
            # This has been modified to reduce computational requirements such that adding a new waveform
            # doesn't require you to recalculate everything

            hd_sort = HD_Sort(trace_source= self.norm_proj,model = self.model, bnum = bnum ,knum = key)
            sort_proj = hd_sort.preprocess()
            
            gal = Group_and_Label(sort_proj, arg_index = hd_sort.HDs_index)
            target_proj = gal.preprocess()
            
            #noize needed to be add here
            
            ae = Auto_Encorder(self.norm_proj, target_proj)
            ae.epoch = self.parameters._epoch
            ae.batch_size = self.parameters._batch_size
            ae_proj = ae.run()
            
            data = []
            textins = []
            textouts = []
            knownkeys = []
            for i in range(tstart, tend):
                    # Handle Offset
                    tnum = i + tracerange[0]

                    try:
                        data.append(ae_proj._traceManager.get_trace(tnum))
                        textins.append(ae_proj._traceManager.get_textin(tnum))
                        textouts.append(ae_proj._traceManager.get_textout(tnum))
                        knownkeys.append(ae_proj._traceManager.get_known_key(tnum))
                    except Exception as e:
                        if progressBar:
                            progressBar.abort(str(e))
                        return

            traces = np.array(data)
            textins = np.array(textins)
            textouts = np.array(textouts)
        
            #Initialize arrays & variables to zero
            sumnum = np.zeros(len(traces[0,:]))

            hyp = [0] * numtraces
            
            if pointRange == None:
                traces = traces_all
            else:
                traces = traces_all[:, pointRange[0] : pointRange[1]]
                
            for tnum in range(numtraces):

                if len(plaintexts) > 0:
                    pt = plaintexts[tnum]

                if len(ciphertexts) > 0:
                    ct = ciphertexts[tnum]

                if knownkeys and len(knownkeys) > 0:
                    nk = knownkeys[tnum]
                else:
                    nk = None

                state['knownkey'] = nk
                
                hypint = self.model.leakage(pt, ct, key, bnum, state)

                hyp[tnum] = hypint

            hyp = np.array(hyp)
            
            #Sumden1/Sumden2 are variance of these variables, may be numeric unstability
            #See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for online update
            #algorithm which might be better

            self.square_sum_trace += np.sum(np.square(traces), axis=0, dtype=np.longdouble)
            self.sum_trace += np.sum(traces, axis=0, dtype=np.longdouble)
            sumden2 = np.square(self.sum_trace) - self.totalTraces * self.square_sum_trace
            
            self.sum_hyp[key] += np.sum(hyp, axis=0, dtype=np.longdouble)
            self.sum_ht[key] += np.sum(np.multiply(np.transpose(traces), hyp), axis=1, dtype=np.longdouble)

            sumnum = self.totalTraces * self.sum_ht[key] - self.sum_hyp[key] * self.sum_trace

            self.square_sum_hyp[key] += np.sum(np.square(hyp),axis=0, dtype=np.longdouble)
            sumden1 = (np.square(self.sum_hyp[key]) - self.totalTraces * self.square_sum_hyp[key])
            sumden = sumden1 * sumden2

            diffs[key] = sumnum / np.sqrt(sumden)

            if progressBar:
                progressBar.updateStatus(pbcnt, (self.totalTraces-numtraces, self.totalTraces-1, bnum))
            pbcnt = pbcnt + 1

        return (diffs, pbcnt)


class LAE_CPA(AlgorithmsBase):
    """
    CPA Attack done as a loop, but using an algorithm which can progressively add traces & give output stats
    """
    _name = "Progressive"

    def __init__(self):
        AlgorithmsBase.__init__(self)

        self.getParams().addChildren([
            {'name':'Iteration Mode', 'key':'itmode', 'type':'list', 'values':{'Depth-First':'df', 'Breadth-First':'bf'}, 'value':'bf', 'action':self.updateScript},
            {'name':'Skip when PGE=0', 'key':'checkpge', 'type':'bool', 'value':False, 'action':self.updateScript},
        ])
        self.updateScript()
    

    def addTraces(self, project:Project, tracerange, parameters, progressBar=None, pointRange=None):
        self._project = project
        self._parameters = parameters
        
        numtraces = tracerange[1] - tracerange[0] + 1
        
        if progressBar:
            progressBar.setText("Attacking traces subset: from %d to %d (total = %d)" % (tracerange[0], tracerange[1], numtraces))
            progressBar.setStatusMask("Trace Interval: %d-%d. Current Subkey: %d")
            progressBar.setMaximum(len(self.brange) * self.model.getPermPerSubkey() * math.ceil(float(numtraces) / self._reportingInterval) - 1)

        pbcnt = 0
        cpa = [None]*(max(self.brange)+1) # None * 16
        for bnum in self.brange:
            cpa[bnum] = CPA_LAEOneSubkey(self.model, self._project, self._parameters)
            
        # what is brangemap doing?
        brangeMap = [None]*(max(self.brange)+1)
        i = 1
        for bnum in self.brange:
            brangeMap[bnum] = i
            i += 1

        skipPGE = False  # self.findParam('checkpge').getValue()
        bf = True  # self.findParam('itmode').getValue() == 'bf'

        # bf specifies a 'breadth-first' search. bf means we search across each subkey by only the amount of traces specified. 
        # Depth-First means to search each subkey completely, then move onto the next.
        
        if bf:
            brange_df = [0]
            brange_bf = self.brange
        else:
            brange_bf = [0] 
            brange_df = self.brange

        for bnum_df in brange_df:
            tstart = 0
            tend = self._reportingInterval

            while tstart < numtraces:
                if tend > numtraces:
                    tend = numtraces

                if tstart > numtraces:
                    tstart = numtraces

                print(tstart, numtraces)
                data = []
                textins = []
                textouts = []
                knownkeys = []
                for i in range(tstart, tend):
                    # Handle Offset
                    tnum = i + tracerange[0]
                    # try:
                    data.append(project._traceManager.get_trace(tnum))
                    textins.append(project._traceManager.get_textin(tnum))
                    textouts.append(project._traceManager.get_textout(tnum))
                    knownkeys.append(project._traceManager.get_known_key(tnum))
                    # except Exception as e:
                    #     if progressBar:
                    #         progressBar.abort(str(e))
                    #     return

                traces = np.array(data)
                textins = np.array(textins)
                textouts = np.array(textouts)
                knownkeys = np.array(knownkeys)

                for bnum_bf in brange_bf:
                    if bf:
                        bnum = bnum_bf
                    else:
                        bnum = bnum_df

                    skip = False
                    # self.stats is defined at analyzer/attacks/cpa_lea.py l:168
                    if (self.stats.simple_PGE(bnum) != 0) or (skipPGE == False): # type: ignore
                        bptrange = pointRange
                        #ここでトレースを渡すとAEを実行する上で色々まずい
                        cpa[bnum].preprocess()
                        (data, pbcnt) = cpa[bnum].oneSubkey(bnum, bptrange, traces, tend, tstart, textins, textouts, knownkeys, progressBar, cpa[bnum].modelstate, pbcnt, tracerange)
                        self.stats.update_subkey(bnum, data, tnum=tend) # type: ignore
                    else:
                        skip = True

                    if skip:
                        pbcnt = brangeMap[bnum] * self.model.getPermPerSubkey() * (numtraces / self._reportingInterval + 1)

                        if bf is False:
                            tstart = numtraces

                    if progressBar and progressBar.wasAborted():
                        return

                tend += self._reportingInterval
                tstart += self._reportingInterval

                if self.sr:
                    self.sr()
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from typing import Optional
from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.analyzer.attacks.models.AES128_8bit import AES128_8bit
from tqdm import trange

from cw_ml_plugin.analyzer.preprocessing.poi_slice import Slice
from cw_ml_plugin.analyzer.preprocessing.convolve import Convolve
from cw_ml_plugin.analyzer.preprocessing.normalize import Normalize
from cw_ml_plugin.analyzer.preprocessing.hd_sort import HD_Sort
from cw_ml_plugin.analyzer.preprocessing.group_and_label import Group_and_Label
from cw_ml_plugin.analyzer.preprocessing.autoencorder import Auto_Encorder



class LAE():
    """
    Attack done by Leakage AutoEncorder and CPA Progressive
    """
    
    __name = "Leakage Auto-Encorder"
    def __init__(self, project: Project, leakage_model:AES128_8bit, epoch , batch_size,
                 convolve_vector = None):
        self._project:Project = project
        self._group_num: int = 1
        if not isinstance(leakage_model, AES128_8bit):
            raise TypeError(f'expected chipwhisperer leakage model: got{type(leakage_model)}')
        else:
            self._leakage_model: AES128_8bit = leakage_model
        self._point_range:tuple = (0,project._traceManager.num_points())
        self._trace_range:tuple = (0,project._traceManager.num_traces())
        self._trace_num:int = project._traceManager.num_points()
        self._trace_length:int = project._traceManager.num_traces()
        self._epoch:int = epoch
        self._batch_size:int = batch_size  
        self._convolve_vector:Optional[np.ndarray] = convolve_vector
        self._update_interval:int = 1000
        
    def _set_group_num(self, gnum:int) -> None:
        if not isinstance(gnum, int):
            raise TypeError(f'expected int; got {type(gnum)}')
        else:
            self._group_num = gnum
            
    @property
    def group_num(self) -> int:
        return self._group_num
    
    @group_num.setter
    def group_num(self, gnum:int):
        self._set_group_num(gnum)
    
    @property
    def leakage_model(self) -> AES128_8bit:
        return self._leakage_model
    
    def _set_convolve_vector(self, v:np.ndarray) -> None:
        if not isinstance(v, np.ndarray):
            raise TypeError(f'expected numpy array; got {type(v)}')
        self._convolve_vector = v
        
    @property
    def convolve_vector(self) -> Optional[np.ndarray]:
        return self._convolve_vector
    
    @convolve_vector.setter
    def convolve_vector(self, v:np.ndarray) -> None:
        self._set_convolve_vector(v)
    
    def _set_point_range(self, range:tuple) -> None:
        if not isinstance(range, tuple):
            raise TypeError(f'expected tuple; got {type(range)}')
        else:
            self._point_range = range
            self._trace_length = range[1] - range[0]
            
    @property
    def point_range(self) -> tuple:
        return self._point_range
    
    @point_range.setter
    def point_range(self, range:tuple):
        self._set_point_range(range)
    
    def _set_trace_range(self, range:tuple) -> None:
        if not isinstance(range, tuple):
            raise TypeError(f'expected tuple; got {type(range)}')
        else:
            self._trace_range = range
            self._trace_num = range[1] - range[0]
            
    @property
    def trace_range(self) -> tuple:
        return self._trace_range
    
    @trace_range.setter
    def trace_range(self, range:tuple) -> None:
        self._set_trace_range(range)
    
    @property
    def trace_length(self) -> int:
        return self._trace_length
    
    @property
    def trace_num(self) -> int:
        return self._trace_num
    
    @property
    def correlations(self) -> np.ndarray:
        return self._correlations
    
    def set_update_interval(self, interval:int) -> None:
        if not isinstance(interval, int):
            raise TypeError(f'expected int; got {type(interval)}')
        else:
            self._update_interval = interval
            
    @property
    def update_interval(self) -> int:
        return self._update_interval
    
    @update_interval.setter
    def update_interval(self, interval:int) -> None:
        self.set_update_interval(interval)
    
    def run(self):
        slice = Slice(self._project)
        slice.poi = self._point_range
        slice.trace_num = self._trace_range
        slice_proj = slice.preprocess()
        
        if self._convolve_vector:
            convolve = Convolve(slice_proj)
            convolve.convolve_mode = 'same'
            convolve.weight_vector = self._convolve_vector
            slice_proj = convolve.preprocess()

        norm = Normalize(slice_proj)
        norm_proj = norm.preprocess()
        
        self._correlations:np.ndarray = np.empty((16,256,slice_proj._traceManager.num_points()),dtype = np.float64)
                
        trace_division = np.arange(start=0, stop=self.trace_length, step=self.update_interval)
        
        for nbyte in trange(16):
            for nkey in trange(256):
                
                hd_sort = HD_Sort(trace_source=norm_proj,model = self._leakage_model, bnum = nbyte ,knum = nkey)
                sort_proj = hd_sort.preprocess()
                
                gal = Group_and_Label(sort_proj, arg_index = hd_sort.HDs_index)
                target_proj = gal.preprocess()
                
                
                #noize needed to be add here
                
                ae = Auto_Encorder(norm_proj, target_proj)
                ae.epoch = self._epoch
                ae.batch_size = self._batch_size
                ae_proj = ae.run()
                
                osubkey = onesubkey(ae_proj, self.leakage_model)
                
                for i in range(len(trace_division)):
                    tstart = trace_division[i]
                    if trace_division[i] != trace_division[-1]:
                        tend = trace_division[i+1]
                    else:
                        tend = self.trace_length

                    self.correlations[nbyte][nkey] = osubkey.calculate(nbyte=nbyte, nkey=nkey, tstart=tstart, tend=tend)
        
        return self.correlations
                
class onesubkey():
    def __init__(self, project:Project, model:AES128_8bit):
        trace_points = project._traceManager.num_points()
        self.model:AES128_8bit = model
        # Doesn't relie on nbkey or nbyte
        self.square_sum_trace:np.ndarray= np.zeros(trace_points, dtype=np.longdouble)
        self.sum_trace:np.ndarray= np.zeros(trace_points, dtype=np.longdouble)
        #Relies on nkey or nbyte
        self.square_sum_hyp = np.zeros(256, dtype=np.int64)
        self.sum_hyp = np.zeros(256, dtype=np.int64)
        self.sum_ht:np.ndarray = np.zeros((256, trace_points), dtype=np.longdouble)
        
        self.totalTraces = 0
        self.project:Project = project
        
    def calculate(self, nbyte, nkey, tstart, tend) -> float:
        trace_num = tend - tstart
        self.totalTraces += trace_num
        
        self.trace:np.ndarray = np.empty((trace_num, self.project._traceManager.num_points()), dtype=np.float64)
        self.pt:np.ndarray = np.empty((trace_num, 16), dtype=np.uint8)
        self.ct:np.ndarray = np.empty((trace_num, 16), dtype=np.uint8)
        
        for tnum in range(tstart,tend):
            self.trace[tnum - tstart] = self.project._traceManager.get_trace(tnum)
            self.pt[tnum - tstart] = self.project._traceManager.get_textin(tnum)
            self.ct[tnum - tstart] = self.project._traceManager.get_textout(tnum)
        
        self.hyp = np.empty(trace_num, dtype=np.uint8)
        for tnum in range(trace_num):
            pt = self.pt[tnum]
            ct = self.ct[tnum]
            hypint = self.model.leakage(pt, ct, nkey, nbyte, {})
            self.hyp[tnum] = hypint
        
        self.square_sum_trace += np.sum(np.square(self.trace), axis=0, dtype=np.longdouble)
        self.sum_trace += np.sum(self.trace, axis=0, dtype=np.longdouble)
        sumden2 = np.square(self.sum_trace) - self.totalTraces * self.square_sum_trace
        
        self.sum_hyp[nkey] += np.sum(self.hyp, axis=0, dtype=np.longdouble)
        self.sum_ht[nkey] += np.sum(np.multiply(np.transpose(self.trace), self.hyp), axis=1, dtype=np.longdouble)
        
        sumnum = self.totalTraces * self.sum_ht[nkey] - self.sum_hyp[nkey] * self.sum_trace
        
        self.square_sum_hyp[nkey] += np.sum(np.square(self.hyp),axis=0, dtype=np.longdouble)
        sumden1 = np.square(self.sum_hyp[nkey]) - (self.totalTraces * self.square_sum_hyp[nkey])
        sumden = sumden1 * sumden2
        
        corr = sumnum / np.sqrt(sumden)
        
        return corr
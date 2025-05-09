import numpy as np
from typing import Optional
from cw_ml_plugin.analyzer.preprocessing._sort_base import SortingBase

class HD_Sort(SortingBase):
    def __init__(self, model, knum, bnum, trace_source = None, name = None): # type: ignore
        SortingBase.__init__(self, trace_source, name = name)
        self._leakage_model = model
        self._trace_length = self.num_points()
        self._trace_num = self.num_traces()
        self._knum: int = knum # key number intrested in
        self._bnum: int = bnum # byte number intrested in
        self._HDs: np.ndarray
        self._HDs_index: np.ndarray

    def _set_knum(self, knum: int) -> None:
        self._knum = knum
    
    @property
    def knum(self) -> Optional[int]:
        return self._knum
    
    @knum.setter
    def knum(self, trace_num: int):
        """Set the trace number you are intreseted in"""
        if not isinstance(trace_num, int):
            raise TypeError(f'expected int; got{type(trace_num)}')
        self._set_knum(trace_num)
        
    def _set_bnum(self, bnum: int) -> None:
        self._bnum = bnum
    
    @property
    def bnum(self) -> Optional[int]:
        return self._bnum
    
    @bnum.setter
    def bnum(self, byte_num: int):
        """Set the byte number you are intreseted in"""
        if not isinstance(byte_num, int):
            raise TypeError(f'expected int; got{type(byte_num)}')
        self._set_bnum(byte_num)
        
    @property
    def leakage_model(self):
        return self._leakage_model
    
    @property
    def trace_length(self):
        return self._trace_length
    
    @property
    def trace_num(self):
        return self._trace_num
    
    @property
    def HDs(self):
        return self._HDs
    
    @property
    def HDs_index(self):
        return self._HDs_index

    def calculate_hd(self, tnum:int, bnum:int, knum:int):
        model = self.leakage_model
        pt = self.get_textin(tnum) 
        ct = self.get_textout(tnum)
        return model.leakage(pt, ct, knum, bnum, {}) 
        
    def sort(self) -> np.ndarray:
        arg_hd = np.empty([self.trace_num,2], dtype = np.uint32)
        arg_hd[:,0] = np.arange(self.trace_num)
        for tnum in range(self.trace_num):
            hd = self.calculate_hd(tnum, self._bnum, self._knum)
            arg_hd[tnum,1] = hd
        arg_hd = arg_hd[np.argsort(arg_hd[:,1])]
        self._HDs, self._HDs_index= np.unique(arg_hd[:,1], return_index = True)
        return arg_hd[:,0]
    
    #sample test code            
    # a = np.empty([10,2], dtype = np.uint32)
    # a[:,0] = np.arange(10)
    # a[:,1] = np.random.randint(low = 0, high = 10, size = 10)
    # print(a)
    # a = a[np.argsort(a[:,1])]
    # print(a[:,0])
    # print(a)
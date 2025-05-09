import numpy as np
import logging
import chipwhisperer as cw
from typing import Optional
from chipwhisperer.analyzer.preprocessing._base import PreprocessingBase
from chipwhisperer.common.api.ProjectFormat import Project
from tqdm import trange

class Group_and_Label(PreprocessingBase):
    """
        trace_source : CW Project that includes the trace data you want to analize
        arg_index[Optional] : Give the head index for each group. This will allow to differ the number of traces by groups.  
    """
    def __init__(self, trace_source = None, name = None, arg_index:Optional[np.ndarray] = None):
       PreprocessingBase.__init__(self, trace_source, name = name)
       self._trace_length:int = self.get_trace_source().num_points() # type: ignore
       self._trace_num: int = self.num_traces()
       if type(arg_index) == np.ndarray:
           self._group_num = len(arg_index)
       else:
           self._group_num: int = 0
       self._arg_index: Optional[np.ndarray] = arg_index

    def _set_group_num(self, group_num:int) -> None:
        self._group_num = group_num
    
    @property
    def group_num(self) -> Optional[int]:
        return self._group_num
    
    @group_num.setter
    def group_num(self, n:Optional[int]) -> None:
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got{type(n)}')
        if self._arg_index != None:
            if n != len(self._arg_index):
                raise ValueError(f'The given value {n} dosent match with the group number given in arg_index{self._arg_index}')
        self._set_group_num(n)
    
    def _make_trace_matrix(self) -> np.ndarray:
        data_matrix = np.empty((self._trace_num, self._trace_length))  
        for trace_num in range(self._trace_num):
            data_matrix[trace_num] = self._traceSource.get_trace(trace_num) # type: ignore
        return data_matrix
    
    def make_label(self) -> np.ndarray:
        if self._group_num == 0:
            raise ValueError(f'Set a group number')
        self._group_matrix: np.ndarray = np.empty(shape = (self._group_num, self._trace_length))
        data = self._make_trace_matrix()
        
        if type(self._arg_index) == np.ndarray:
            self._head_index = self._arg_index.astype(np.uint16)
            self._head_index = np.append(self._head_index, self._trace_num)
        else:
            self._head_index = np.linspace(start=0, stop = self._trace_num, num = self._group_num + 1, dtype=np.uint64)

        for gnum in range(self._group_num):
            matrix = data[self._head_index[gnum]:self._head_index[gnum+1],:]
            mean = np.mean(matrix,axis=0)
            self._group_matrix[gnum] = mean
        
        return self._group_matrix
        
    def preprocess(self) -> Project:
        self.make_label()
        proj = Project()
        
        gnum = 0
        for tnum in range(self.num_traces()):
            if tnum == self._head_index[gnum+1]:
                gnum = gnum + 1
            
            proj.traces.append(cw.Trace(self._group_matrix[gnum], self.get_textin(tnum), self.get_textout(tnum), self.get_known_key(tnum))) # type: ignore
        return proj
import numpy as np
import chipwhisperer as cw
from tqdm import tqdm

from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.analyzer.preprocessing._base import PreprocessingBase

class Normalize(PreprocessingBase):
    def __init__(self, trace_source = None, name = None, minus_avg = True):
        PreprocessingBase.__init__(self, trace_source, name = name)
        self._trace_num:int = self.num_traces()
        self._trace_length:int = self.get_trace_source().num_points() # type: ignore
        self._minus_avg = minus_avg


    def _make_trace_matrix(self) -> np.ndarray:
        data_matrix = np.empty((self._trace_num, self._trace_length))
        for trace_num in range(self._trace_num):
            data_matrix[trace_num] = self._traceSource.get_trace(trace_num) # type: ignore
        return data_matrix

    def norm(self, trace:np.ndarray) -> np.ndarray:
        avg = np.average(trace, axis=0)
        if self._minus_avg:
            trace -= avg
        x_max = np.max(trace)
        x_min = np.min(trace)
        norm_trace = (trace - x_min) / (x_max-x_min)
        norm_trace = norm_trace*2-1
        return norm_trace

    def preprocess(self) -> Project:
        norm_matrix = self.norm(self._make_trace_matrix())
        project = Project()
        pbar = tqdm(range(self._trace_num))
        for tnum in pbar:
            project.traces.append(cw.Trace(norm_matrix[tnum],self.get_textin(tnum),self.get_textout(tnum),self.get_known_key(tnum))) # type: ignore
        return project
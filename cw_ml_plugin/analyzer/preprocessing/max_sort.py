import numpy as np
from typing import Optional
from cw_ml_plugin.analyzer.preprocessing._sort_base import SortingBase

class Max_Sort(SortingBase):
    def __init__(self, trace_source=None, name = None):
        SortingBase.__init__(self, trace_source, name = name)
        self._arg = self.sort()

    def _make_trace_matrix(self) -> np.ndarray:
        if self._traceSource.get_trace(0) is None: # type: ignore
            raise ValueError(f'Trace Data Unfound')
        else:
            trace_length = len(self._traceSource.get_trace(0)) # type: ignore

        trace_matrix = np.empty((self.num_traces(),trace_length))

        for trace_num in range(self.num_traces()):
            trace_matrix[trace_num] = self._traceSource.get_trace(trace_num) # type: ignore
        return trace_matrix

    def sort(self) -> np.ndarray:
        trace_max = np.max(self._make_trace_matrix(), axis = 1)
        arg = np.argsort(trace_max)
        return arg

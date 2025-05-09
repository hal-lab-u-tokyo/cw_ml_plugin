import numpy as np
from scipy.stats import norm
from typing import Literal, Optional
from chipwhisperer.analyzer.preprocessing._base import PreprocessingBase

class AddNoize(PreprocessingBase):
    def __init__(self, trace_source=None, name = None):
        PreprocessingBase.__init__(self, trace_source, name = name)
        self._noize_range: Optional[np.ndarray] = None

    def _set_noize_range(self, noize_range):
        self._noize_range = noize_range
        self._param, self._noizy_trace = self._estimate_noize_model(noize_range)
        self.proc = np.empty((self.num_traces(), self.get_trace_source().num_points())) # type: ignore

    @property
    def noize_range(self):
        return self._noize_range

    @noize_range.setter
    def noize_range(self, noize_range:list):
        """Set the nozie_range to estimate noize models"""
        if not isinstance(noize_range, list):
            raise TypeError(f'Expected list; got {type(noize_range)}')

        self._set_noize_range(noize_range)

    def _make_trace_matrix(self) -> np.ndarray:
        if self._traceSource.get_trace(0) is None: # type: ignore
            raise ValueError(f'Trace Data Unfound')
        else:
            trace_length = len(self._traceSource.get_trace(0)) # type: ignore

        trace_matrix = np.empty((self.num_traces(),trace_length))

        for trace_num in range(self.num_traces()):
            trace_matrix[trace_num] = self._traceSource.get_trace(trace_num) # type: ignore
        return trace_matrix

    def _estimate_noize_model(self, noize_range):
        trace_noize = self._make_trace_matrix()[:, noize_range[0]:noize_range[1]]
        # 平均を取る
        med = np.mean(trace_noize, axis=0)

        # 全体から引く
        trace_noize -= med

        # 配列を1次元にフラット化
        trace_noize = trace_noize.flatten()

        # ガウシアンフィッティングを行い、パラメータを取得
        param = norm.fit(trace_noize)
        shape = (self.num_traces(), self.get_trace_source().num_points()) # type: ignore
        noizy_trace = np.random.normal(loc=param[0], scale=param[1], size=shape)
        return param, noizy_trace


    def get_trace(self, n: int) -> Optional[np.ndarray]:
        if self.enabled:
            trace = self._traceSource.get_trace(n) # type: ignore

            if trace is None:
                return None

            return trace + self._noizy_trace[n]
        else:
            return self._traceSource.get_trace(n) # type: ignore
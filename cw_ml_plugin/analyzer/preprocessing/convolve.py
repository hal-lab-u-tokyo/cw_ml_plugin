import numpy as np
from typing import Literal, Optional
from chipwhisperer.analyzer.preprocessing._base import PreprocessingBase

class Convolve(PreprocessingBase):
    def __init__(self, trace_source=None, name = None):
        PreprocessingBase.__init__(self, trace_source, name = name)
        self._weight_vector: Optional[np.ndarray] = None
        self._convolve_mode: Literal['full', 'same', 'valid'] = 'same'

    def _set_weight_vector(self, weight_vector):
        self._weight_vector = weight_vector

    @property
    def weight_vector(self):
        return self._weight_vector

    @weight_vector.setter
    def weight_vector(self, array: Optional[np.ndarray]) -> None:
        """Set the weight array for convolution"""
        if not isinstance(array, np.ndarray):
            raise TypeError(f'Expected numpy array; got {type(array)}')
        self._set_weight_vector(array)

    def _set_convolve_mode(self, mode: Literal['full', 'same', 'valid']):
        self._convolve_mode = mode

    @property
    def convolve_mode(self):
        return self._convolve_mode

    @convolve_mode.setter
    def convolve_mode(self, mode: Literal['full', 'same', 'valid']):
        """Set the convolving mode.
           choose from 'full','same','valid'
           for futher information look https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        """
        if not (mode == 'full' or mode == 'same' or mode == 'valid'):
            raise TypeError(f'Expected string; got {type(mode)}')
        return self._set_convolve_mode(mode)

    def _convolve(self, input_trace: np.ndarray) -> np.ndarray:
        output_trace = np.convolve(input_trace, self._weight_vector, self._convolve_mode)  # type: ignore
        return output_trace

    def get_trace(self, n: int) -> Optional[np.ndarray]:
        if self.enabled:
            trace = self._traceSource.get_trace(n) # type: ignore

            if trace is None:
                return None

            proc = self._convolve(trace)

            return proc
        else:
            return self._traceSource.get_trace(n) # type: ignore
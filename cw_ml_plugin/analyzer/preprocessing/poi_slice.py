import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from typing import Optional
from chipwhisperer.common.api.ProjectFormat import Project

class Slice():
    """
        poi(list): The point of range in the trace
        trace_num(list) = (start,end): The traces that will be used for analysis. Traces in [start,end)
    """
    def __init__(self, project: Project):
        self._project: Project = project
        self._poi: list = [0, project._traceManager.num_points()]
        self._trace_num: list = [0,len(project.traces)]

    def _set_poi(self, poi:list) -> None:
        self._poi = poi

    @property
    def poi(self) -> Optional[list]:
        if self._poi == (0,-1):
            print("POI unsetted")
            return None
        else:
            return self._poi

    @poi.setter
    def poi(self, rng:list) -> None:
        if not isinstance(rng, list):
            raise TypeError(f'expected list; got {type(rng)}')
        self._set_poi(rng)

    def _set_trace_num(self, trace_num: list) -> None:
        self._trace_num = trace_num

    @property
    def trace_num(self) -> list:
        return self._trace_num

    @trace_num.setter
    def trace_num(self, num: list):
        if not isinstance(num, list):
            raise TypeError(f'expected int; got {type(num)}')
        self._set_trace_num(num)

    def preprocess(self) -> Project:
        new_project = Project()
        for num in range(self._trace_num[0], self._trace_num[1]):
            value = self._project.waves[num][self._poi[0]:self._poi[1]]
            plain = self._project.textins[num]
            encrypt = self._project.textouts[num]
            key = self._project.keys[num]
            new_project.traces.append(cw.Trace(value, plain, encrypt, key)) # type: ignore
        return new_project

from collections.abc import Iterable
from typing import Literal
import numpy as np
import multiprocessing as mp

from chipwhisperer.analyzer.attacks.cpa import CPA as CPA_Old
from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase
from collections import OrderedDict
from chipwhisperer.analyzer.attacks.models.AES128_8bit import AES128_8bit
from chipwhisperer.common.utils.util import dict_to_str

from cw_ml_plugin.analyzer.preprocessing.normalize import Normalize
from cw_ml_plugin.analyzer.preprocessing.poi_slice import Slice
from cw_ml_plugin.analyzer.attacks.algorithms.ddla_algorithm import DDLA_Algorithm
from cw_ml_plugin.common.results.plot import overall


class DDLA(CPA_Old):
    _project = None
    reporting_interval = 10
    # AlgorithmsBase_subclass = TypeVar('AlgorithmsBase_subclass', bound = 'AlgorithmsBase')
    # algorithmの型ヒントについては工夫が必要
    def __init__(self, 
                 proj:Project, 
                 leak_model:AES128_8bit, 
                 epoch: int, 
                 batch_size: int,
                 label_func, 
                 learning_rate= 0.001, 
                 max_paralell_jobs= 1):
        
        self.parameters = parameters()
        
        self.parameters._trace_range = [0,proj._traceManager.num_traces()]
        self.parameters._trace_num = proj._traceManager.num_points()
        self.parameters._trace_length = proj._traceManager.num_traces()
        self.parameters._epoch = epoch
        self.parameters._batch_size = batch_size
        self.parameters._learning_rate = learning_rate
        self.parameters._byte_range = range(16)
        self.parameters._key_range = range(256)
        self.parameters._sort_param = 'accuracy'
        self.parameters._update_interval = 1000
        self.parameters._max_parallel_jobs = max_paralell_jobs
        algorithm = DDLA_Algorithm
        self.label_func = label_func
        
        #need to attribute values to aecpa class also to use the original super().__init__() below
        for attr_name, attr_value in vars(self.parameters).items():
            setattr(self, attr_name, attr_value)

        super().__init__()
        self.set_analysis_algorithm(algorithm, leak_model)
        self.set_trace_source(proj.trace_manager())
        self.change_project(proj)
        self.algorithm = algorithm # type: ignore
        self.leak_model = leak_model
        self.set_iterations(1)

        pass

    def _dict_repr(self):
        dict = OrderedDict()
        dict['project'] = self.project
        dict['leak_model'] = self.leak_model
        dict['algorithm'] = self.algorithm
        dict['trace_range'] = self.trace_range
        dict['point_range'] = self.point_range
        dict['epoch'] = self.epoch
        dict['batch_size'] = self.batch_size
        dict['subkey_list'] = self.subkey_list
        return dict

    def __str__(self):
        return self.__repr__() + '\n' + dict_to_str(self._dict_repr())

    @property
    def algorithm(self) :
        return self.attack

    @algorithm.setter
    def algorithm(self, alg:AlgorithmsBase):
        self.set_analysis_algorithm(alg)

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, proj):
        self._project = proj
        self.set_trace_source(proj.trace_manager())

    @property
    def leak_model(self):
        return self.get_leak_model()

    @leak_model.setter
    def leak_model(self, model):
        self.set_leak_model(model)

    @property
    def point_range(self):
        return self.get_point_range()

    @point_range.setter
    def point_range(self, rng):
        self.set_point_range(rng)
        self.parameters._point_range = rng

    def _set_trace_range(self, range:list) -> None:
        if not isinstance(range, list):
            raise TypeError(f'expected tuple; got {type(range)}')
        else:
            self.parameters._trace_range = range
            self.parameters._trace_num = range[1] - range[0]

            self._trace_range = range
            self._trace_num = range[1] - range[0]

    @property
    def trace_range(self) -> list:
        return self._trace_range

    @trace_range.setter
    def trace_range(self, range:list) -> None:
        self._set_trace_range(range)

    @property
    def subkey_list(self):
        return self.get_target_subkeys()

    @subkey_list.setter
    def subkey_list(self, subkeys):
         self.set_target_subkeys(subkeys)

    def _set_epoch(self, ep:int) -> None:
        if not isinstance(ep, int):
            raise TypeError(f'expected int; got {type(ep)}')
        else:
            self._epoch = ep
            self.parameters._epoch = ep

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, ep:int):
        self._set_epoch(ep)
    
    def _set_batch_size(self, bs:int) -> None:
        if not isinstance(bs, int):
            raise TypeError(f'expected int; got {type(bs)}')
        else:
            self.parameters._batch_size = bs
            self._batch_size = bs
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, bs:int):
        self._set_batch_size(bs)

    def _set_learning_rate(self, lr:float) -> None:
        if not isinstance(lr, float):
            raise TypeError(f'expected float; got {type(lr)}')
        else:
            self._learning_rate = lr
            self.parameters._learning_rate = lr
            
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, lr:float):
        self._set_learning_rate(lr)
    
    def _set_byte_range(self, rng:Iterable) -> None:
        if not isinstance(rng, Iterable):
            raise TypeError(f'expected Iterable; got {type(rng)}')
        else:
            self._byte_range = rng
            self.parameters._byte_range = rng
    
    @property
    def byte_range(self) -> Iterable:
        return self._byte_range
    
    @byte_range.setter
    def byte_range(self, rng:Iterable):
        self._set_byte_range(rng)
        
    def _set_key_range(self, rng:Iterable) -> None:
        if not isinstance(rng, Iterable):
            raise TypeError(f'expected Iterable; got {type(rng)}')
        else:
            self._key_range = rng
            self.parameters._key_range = rng
            
    @property
    def key_range(self) -> Iterable:
        return self.parameters._key_range
    
    @key_range.setter
    def key_range(self, rng:Iterable):
        self._set_key_range(rng)

    def _set_sort_param(self, sp:Literal['loss', 'accuracy', 'sensitivity']) -> None:
        sp_type = {'loss', 'accuracy', 'sensitivity'}
        if not sp in sp_type:
            raise TypeError(f'expected one of {sp_type}; got {sp, type(sp)}')
        else:
            self._sort_param = sp
            self.parameters._sort_param = sp
            
    @property
    def sort_param(self) -> str:
        return self._sort_param
    
    @sort_param.setter
    def sort_param(self, sp:Literal['loss', 'accuracy', 'sensitivity']):
        self._set_sort_param(sp)

    def _set_max_parallel_jobs(self, mpj:int) -> None:
        if not isinstance(mpj, int):
            raise TypeError(f'expected int; got {type(mpj)}')
        else:
            self._max_parallel_jobs = mpj
            self.parameters._max_parallel_jobs = mpj
    
    @property
    def max_parallel_jobs(self) -> int:
        return self._max_parallel_jobs
    
    @max_parallel_jobs.setter
    def max_parallel_jobs(self, mpj:int):
        self._set_max_parallel_jobs(mpj)
        
    def change_project(self, proj):
        """ Change the project property and update ranges

        If you don't want to update your ranges, change the project attribute
        instead.

        Args:
            project (Project): Project to switch to

        Returns:
            None
        """
        self.project = proj
        self.trace_range = [0, len(proj.traces)]
        self.point_range = [0, len(proj.waves[0])]
        self.subkey_list = range(0, len(proj.keys[0]))

    @property
    def results(self):
        return self.get_statistics()

    def run(self, callback=None, update_interval=25):
        slice = Slice(self.project) # type: ignore
        slice.poi = self.parameters._point_range
        slice.trace_num = self.parameters._trace_range
        slice_proj = slice.preprocess()

        norm = Normalize(slice_proj, minus_avg=True)
        self.norm_proj = norm.preprocess()
        
        # overall(self.norm_proj, "DDLA inputs", save= False)
        
        if update_interval:
            self.reporting_interval = update_interval
        
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        
        self.algorithm.setModel(self.leak_model) # type: ignore
        self.algorithm.get_statistics().clear() # type: ignore
        self.algorithm.set_reporting_interval(self.reporting_interval) # type: ignore
        self.algorithm.set_target_subkeys(self.get_target_subkeys()) # type: ignore
        self.algorithm.setStatsReadyCallback(callback) # type: ignore
        self.algorithm.addTraces(self.norm_proj, self.parameters, self.label_func)  # type: ignore
        return self.results

class parameters():

    def __init__(self):
        self._point_range: list
        self._trace_range: list 
        self._trace_num: int 
        self._trace_length:int 
        self._epoch: int 
        self._batch_size: int 
        self._learning_rate: float
        self._byte_range: Iterable = range(16)
        self._key_range: Iterable = range(256)
        self._sort_param: Literal['loss', 'accuracy', 'sensitivity']
        self._update_interval:int = 1000
        self._max_parallel_jobs: int
from typing import Optional, Type, TypeVar
import numpy as np

from chipwhisperer.analyzer.attacks.cpa import CPA as CPA_Old
from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase 
from collections import OrderedDict
from chipwhisperer.analyzer.attacks.models.AES128_8bit import AES128_8bit
from chipwhisperer.common.utils.util import dict_to_str

from cw_ml_plugin.analyzer.attacks.algorithms.lae_cpa import LAE_CPA
class AECPA(CPA_Old):
    _project = None
    reporting_interval = 10
    # AlgorithmsBase_subclass = TypeVar('AlgorithmsBase_subclass', bound = 'AlgorithmsBase')
    # algorithmの型ヒントについては工夫が必要
    def __init__(self, proj:Project, leak_model:AES128_8bit, epoch , batch_size,
                 algorithm= LAE_CPA, group_num= 1, convolve_vector= None):
        """
        Args:
            proj (Project): chipwhisperer Project
            leak_model (AESLeakageHelper): Leakage model to use for getting
                hamming weight
            algorithm (AlgorithmsBase): Algorithm to use for attack
            epoch: epoch number for the AE
            batch_size: batch_size for the AE
        """
        self.parameters = parameters()
        
        self.parameters._trace_range = [0,proj._traceManager.num_traces()]
        self.parameters._trace_num = proj._traceManager.num_points()
        self.parameters._trace_length = proj._traceManager.num_traces()
        self.parameters._epoch = epoch
        self.parameters._batch_size = batch_size
        self.parameters._group_num = group_num
        self.parameters._convolve_vector = convolve_vector
        self.parameters._update_interval = 1000
        
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
        dict['group_num'] = self.group_num
        dict['convolve_vector'] = self.convolve_vector
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
    def trace_num(self) -> int:
        return self.trace_num
    
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

    def _set_batch_size(self, bs:int) -> None:
        if not isinstance(bs, int):
            raise TypeError(f'expected int; got {type(bs)}')
        else:
            self.parameters._batch_size = bs
    @property
    def batch_size(self) -> int:
        return self.parameters._batch_size
    
    @batch_size.setter
    def batch_size(self, bs:int):
        self._set_batch_size(bs)

    def _set_epoch(self, ep:int) -> None:
        if not isinstance(ep, int):
            raise TypeError(f'expected int; got {type(ep)}')
        else:
            self._epoch = ep
            
    @property
    def epoch(self) -> int:
        return self._epoch
    
    @epoch.setter
    def epoch(self, ep:int):
        self._set_epoch(ep)

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

    def _set_convolve_vector(self, cvector:np.ndarray) -> None:
        if not isinstance(cvector, np.ndarray):
            raise TypeError(f'expected np array; got {type(cvector)}')
        else:
            self._convolve_vector = cvector
            
    @property
    def convolve_vector(self) -> np.ndarray:
        return self._convolve_vector # type: ignore
        
    @convolve_vector.setter
    def convolve_vector(self, cvector:np.ndarray):
        self._set_convolve_vector(cvector)
    
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
        """ Runs the attack

        Args:
            callback (function(), optional): Callback to call every update
                interval. No arguments are passed to callback. Defaults to None.
            update_interval (int, optional):  Number of traces to process
                before updating the results of the attack.

        Returns:
            Results, the results of the attack. See documentation
            for Results for more details.
        """
        if update_interval:
            self.reporting_interval = update_interval
        self.algorithm.setModel(self.leak_model) # type: ignore
        self.algorithm.get_statistics().clear() # type: ignore
        self.algorithm.set_reporting_interval(self.reporting_interval) # type: ignore
        self.algorithm.set_target_subkeys(self.get_target_subkeys()) # type: ignore
        self.algorithm.setStatsReadyCallback(callback) # type: ignore
        self.algorithm.addTraces(project= self.project, tracerange = self.trace_range, parameters = self.parameters, # type: ignore
                                            pointRange=self.point_range) 
        return self.results

class parameters():

    def __init__(self):
        self._point_range: list
        self._trace_range: list 
        self._trace_num: int 
        self._trace_length:int 
        self._epoch: int 
        self._batch_size: int 
        self._group_num: int 
        self._convolve_vector: Optional[np.ndarray]
        self._update_interval:int = 1000
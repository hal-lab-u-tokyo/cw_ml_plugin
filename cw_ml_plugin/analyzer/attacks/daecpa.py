from typing import Optional, Type, TypeVar
import numpy as np

from chipwhisperer.analyzer.attacks.cpa import CPA as CPA_Old
from chipwhisperer.analyzer.attacks.cpa_algorithms.progressive import CPAProgressive
from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase 
from collections import OrderedDict
from chipwhisperer.analyzer.attacks.models.AES128_8bit import AES128_8bit
from chipwhisperer.common.utils.util import dict_to_str

from cw_ml_plugin.analyzer.attacks.algorithms.lae_cpa import LAE_CPA

from cw_ml_plugin.analyzer.preprocessing.poi_slice import Slice
from cw_ml_plugin.analyzer.preprocessing.convolve import Convolve
from cw_ml_plugin.analyzer.preprocessing.normalize import Normalize
from cw_ml_plugin.analyzer.preprocessing.max_sort import Max_Sort
from cw_ml_plugin.analyzer.preprocessing.group_and_label import Group_and_Label
from cw_ml_plugin.analyzer.preprocessing.autoencorder import Auto_Encorder
from cw_ml_plugin.analyzer.preprocessing.addnoize import AddNoize

class DAECPA(CPA_Old):
    """Correlation Power Analysis (CPA).

    Provides all the needed functionality for taking a project and performing
    a CPA attack with a specific type of leakage model.

    Args:
        proj (Project): An open project instance.
        algorithm: The algorithm used for the analysis. Default is Progressive
            which allows a callback to be given. The callback is called at
            increments of a specific interval. This is useful for auto-updating
            tables and equivalent.
        leak_model: The leakage model used when analysing the captured traces.

    The easiest way to use this class is through the
    :func:`cpa <chipwhisperer.analyzer.cpa>` API function provided by the
    :mod:`chipwhisperer.analyzer` module.

    Example::

        import chipwhisperer.analyzer as cwa
        import chipwhisperer as cw
        proj = cw.open_project('/path/to/project')
        attack = cwa.cpa(proj, cwa.leakage_models.sbox_output)
        results = attack.run()
        print(results)

    Attributes:
        project: Project to pull waves, textin, etc. from.
        algorithm: Analysis algorithm to use for attack. Should be Progressive
        leak_model: Leakage model to use during analysis. Should be of
            type AESLeakageHelper.
        trace_range: Start and end trace number. Should be a list of length 2
            (i.e. [start_num, end_num]).
        point_range: Range of points to use from waves in project. Should be
            a list of length 2 ([start_point, end_point]).
        subkey_list: List of subkeys to attack (subkey_list = [0, 1, 3] will
            attack subkeys 0, 1, and 3).

    .. versionadded:: 5.1
        Added CPA in cpa_new.py to wrap old CPA object
    """
    _project = None
    reporting_interval = 10
    # AlgorithmsBase_subclass = TypeVar('AlgorithmsBase_subclass', bound = 'AlgorithmsBase')
    # algorithmの型ヒントについては工夫が必要
    def __init__(self, proj:Project, leak_model:AES128_8bit, epoch , batch_size,
                 algorithm=CPAProgressive, group_num= 1, convolve_vector= None, noize_range= None):
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
        self.parameters._trace_num = proj._traceManager.num_traces()
        self.parameters._epoch = epoch
        self.parameters._batch_size = batch_size
        self.parameters._group_num = group_num
        self.parameters._convolve_vector = convolve_vector
        self.parameters._noize_range = noize_range
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
        
    def _set_noize_range(self, range:list):
        if not isinstance(range, list):
            raise TypeError(f'expected list; got {type(range)}')
        else:
            self._noize_range = range
            
    @property
    def noize_range(self):
        return self._noize_range
    
    @noize_range.setter
    def noize_range(self, range):
        self._set_noize_range(range)
    
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
        
        #Denozie Auto Encorder        
        proj = self.project
        
        if isinstance(self.convolve_vector, np.ndarray):
            print("CONVOLVING")
            convolve = Convolve(proj)
            convolve.convolve_mode = 'valid'
            convolve.weight_vector = self._convolve_vector
            convolve_proj = convolve.preprocess()
        else:
            convolve_proj = proj

        if self._noize_range != None:
            print("ADDING NOIZE")
            add_noize = AddNoize(convolve_proj)
            add_noize.noize_range = self._noize_range
            addnoize_proj = add_noize.preprocess()
        else:
            addnoize_proj = convolve_proj
        
        print("NORMALIZING")
        norm = Normalize(addnoize_proj, minus_avg = False)
        norm_proj = norm.preprocess()
        
        print("SORTING")
        max_sort = Max_Sort(trace_source=norm_proj)
        sort_proj = max_sort.preprocess()
        
        print("GROUPING AND LABELING")
        gal = Group_and_Label(sort_proj)
        gal.group_num = self.group_num
        target_proj = gal.preprocess()
    
        print("SLICING INPUT TRACES")
        slice = Slice(norm_proj) # type: ignore
        slice.poi = self.parameters._point_range
        slice.trace_num = self._trace_range
        input_proj = slice.preprocess()
        
        print("SLICING TARGET TRACES")
        slice = Slice(target_proj)
        slice.poi = self.parameters._point_range
        slice.trace_num = self._trace_range
        target_proj = slice.preprocess()
        
        print("STARTING AUTO ENCORDER")
        ae = Auto_Encorder(input_proj, target_proj)
        ae.epoch = self._epoch
        ae.batch_size = self._batch_size # type: ignore
        ae_proj = ae.run()
    
        self.project = ae_proj
        
        self.change_project(ae_proj)
        for attr_name, attr_value in vars(self.parameters).items():
            setattr(self, attr_name, attr_value)
        
        print("STARTING CPA")
        if update_interval:
            self.reporting_interval = update_interval
        self.algorithm.setModel(self.leak_model) # type: ignore
        self.algorithm.get_statistics().clear() # type: ignore
        self.algorithm.set_reporting_interval(self.reporting_interval) # type: ignore
        self.algorithm.set_target_subkeys(self.get_target_subkeys()) # type: ignore
        self.algorithm.setStatsReadyCallback(callback) # type: ignore
        self.algorithm.addTraces(self.get_trace_source(), self.trace_range, # type: ignore
                                 None, pointRange=self.point_range) 
        return self.results

class parameters():

    def __init__(self):
        self._point_range: list
        self._trace_range: list 
        self._trace_num: int 
        self._epoch: int 
        self._batch_size: int 
        self._group_num: int 
        self._convolve_vector: Optional[np.ndarray]
        self._noize_range: Optional[list]
        self._update_interval:int = 1000
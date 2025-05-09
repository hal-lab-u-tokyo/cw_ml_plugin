import numpy as np
import chipwhisperer as cw
from typing import Optional

from chipwhisperer.common.api.ProjectFormat import Project
from chipwhisperer.analyzer.attacks.models.AES128_8bit import AES128_8bit

from cw_ml_plugin.analyzer.preprocessing.poi_slice import Slice
from cw_ml_plugin.analyzer.preprocessing.convolve import Convolve
from cw_ml_plugin.analyzer.preprocessing.normalize import Normalize
from cw_ml_plugin.analyzer.preprocessing.max_sort import Max_Sort
from cw_ml_plugin.analyzer.preprocessing.group_and_label import Group_and_Label
from cw_ml_plugin.analyzer.preprocessing.autoencorder import Auto_Encorder
from cw_ml_plugin.analyzer.preprocessing.addnoize import AddNoize

class DAE():
    def __init__(self,
                 project: Project,
                 leak_model: AES128_8bit,
                 epoch: int,
                 batch_size: int,
                 group_num: int = 1,
                 convolve_vector: Optional[np.ndarray] = None,
                 noize_range: Optional[list] = None,
                 hook_layer_num: Optional[int] = None
                 ):
        
        self.parameters = parameters()
        self.parameters._point_range= [0,project._traceManager.num_points()]
        self.parameters._trace_num = project._traceManager.num_traces()
        self.parameters._trace_range = [0, self.parameters._trace_num]
        self.parameters._epoch = epoch
        self.parameters._batch_size = batch_size
        self.parameters._group_num = group_num
        self.parameters._convolve_vector = convolve_vector
        self.parameters._noize_range = noize_range
        self.parameters._update_interval = 1000
        self.parameters._hook_layer_num = hook_layer_num       

        self._project = project
        self._leak_model = leak_model
        
    @property
    def project(self):
        return self._project
        
    def set_leak_model(self, model:AES128_8bit):
        self._leak_model = model 
    
    @property
    def leak_model(self):
        return self._leak_model

    @leak_model.setter
    def leak_model(self, model):
        self.set_leak_model(model)
    
    def set_point_range(self, rng:list):
        if not isinstance(rng, list):
            raise TypeError(f'expected list; got {type(rng)}')
        else:
            self.parameters._point_range = rng
            
    @property
    def point_range(self):
        return self.parameters._point_range

    @point_range.setter
    def point_range(self, rng):
        self.set_point_range(rng)
        self.parameters._point_range = rng

    @property
    def trace_num(self) -> int:
        return self.parameters._trace_num

    def _set_trace_range(self, range:list) -> None:
        if not isinstance(range, list):
            raise TypeError(f'expected tuple; got {type(range)}')
        else:
            self.parameters._trace_range = range
            self.parameters._trace_num = range[1] - range[0]

    @property
    def trace_range(self) -> list:
        return self.parameters._trace_range
    
    @trace_range.setter
    def trace_range(self, range:list) -> None:
        self._set_trace_range(range)

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
            self.parameters._epoch = ep
            
    @property
    def epoch(self) -> int:
        return self.parameters._epoch
    
    @epoch.setter
    def epoch(self, ep:int):
        self._set_epoch(ep)

    def _set_group_num(self, gnum:int) -> None:
        if not isinstance(gnum, int):
            raise TypeError(f'expected int; got {type(gnum)}')
        else:
            self.parameters._group_num = gnum
            
    @property
    def group_num(self) -> int:
        return self.parameters._group_num
    
    @group_num.setter
    def group_num(self, gnum:int):
        self._set_group_num(gnum)

    def _set_convolve_vector(self, vector:np.ndarray) -> None:
        if not isinstance(vector, np.ndarray):
            raise TypeError(f'expected np array; got {type(vector)}')
        else:
            self.parameters._convolve_vector = vector
            
    @property
    def convolve_vector(self) -> np.ndarray:
        return self.parameters._convolve_vector # type: ignore
        
    @convolve_vector.setter
    def convolve_vector(self, vector:np.ndarray):
        self._set_convolve_vector(vector)
        
    def _set_noize_range(self, range:list):
        if not isinstance(range, list):
            raise TypeError(f'expected list; got {type(range)}')
        else:
            self.parameters._noize_range = range
            
    @property
    def noize_range(self):
        return self.parameters._noize_range
    
    @noize_range.setter
    def noize_range(self, range):
        self._set_noize_range(range)
            
    def preprocess(self):
        # trace num needs to be sliced at this point
        # add noize might use traces out of scope
        slice = Slice(self._project)
        slice.trace_num = self.trace_range
        project = slice.preprocess()
        
        print("GENERATING LABELS")
        if isinstance(self.convolve_vector, np.ndarray):
            print("CONVOLVING INPUT")
            convolve = Convolve(project)
            convolve.convolve_mode = 'valid'
            convolve.weight_vector = self.convolve_vector
            convolve_proj = convolve.preprocess()
        else:
            convolve_proj = project
                
        if self.group_num != 0:
            print("SORTING")
            max_sort = Max_Sort(convolve_proj)
            sort_proj = max_sort.preprocess()
            
            print("GROUPING AND LABELING")
            gal = Group_and_Label(sort_proj)
            gal.group_num = self.group_num
            label_proj = gal.preprocess()
        else:
            label_proj = convolve_proj

        print("GENERATING INPUTS")
        if self.noize_range != None:
            print("ADDING NOIZE FOR LEARNING")
            add_noize = AddNoize(project)
            add_noize.noize_range = self.noize_range
            addnoize_proj = add_noize.preprocess()
        else:
            addnoize_proj = project
        
        print("NOMALIZING INPUT TRACES")
        norm = Normalize(addnoize_proj, minus_avg=False)
        norm_input_proj = norm.preprocess()
                
        print("SLICING INPUT TRACES")
        slice = Slice(norm_input_proj)
        slice.poi = self.parameters._point_range
        input_proj = slice.preprocess()
    
        print("NOMALIZING TARGET TRACES")
        norm = Normalize(label_proj, minus_avg=False)
        norm_target_proj = norm.preprocess()
        
        print("SLICING TARGET TRACES")
        slice = Slice(norm_target_proj)
        slice.poi = self.parameters._point_range
        target_proj = slice.preprocess()
        
        print("STARTING AUTO ENCORDER")
        if self.parameters._hook_layer_num != None:
            ae = Auto_Encorder(input_proj, target_proj, hook_layer_num = self.parameters._hook_layer_num)
            ae.epoch = self.epoch
            ae.batch_size = self.batch_size
            ae_proj = ae.run_interpert()
        else:
            ae = Auto_Encorder(input_proj, target_proj)
            ae.epoch = self.epoch
            ae.batch_size = self.batch_size
            ae_proj = ae.run()
        return ae_proj
    
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
        self._leak_model: AES128_8bit
        self._update_interval:int = 1000
        self._hook_layer_num: Optional[int] = None
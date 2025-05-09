import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
import cw_ml_plugin.analyzer as cwma
from chipwhisperer.common.api.ProjectFormat import Project
from cw_ml_plugin.analyzer.attacks.aecpa import AECPA
from chipwhisperer.analyzer.attacks.cpa_algorithms.progressive import CPAProgressive

@pytest.fixture
def base_project() -> Project:
    proj = Project()
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9,   207,  79,  60], dtype = np.uint8)
    plains = np.array([[185, 127, 124, 245,  38, 137, 116,  46,  55, 248,  54, 155,   0, 133, 171, 137],
                       [ 71, 218,  82, 195, 130, 138, 150, 236, 238,  82, 249, 125,  84, 39,   6, 140],
                       [125,  83, 157, 235,  46,  82, 182,  53,  25,  57,  10,  40,  74, 211, 133, 102],
                       [236, 192,  20, 106,  53, 193,  73, 142, 180, 165, 103, 161,  37, 70, 103, 253],
                       [ 57, 138,  95, 104, 232, 181,  97, 127,  56, 205, 225, 125,  98, 249, 170, 180],
                       [ 94,  69, 205, 180, 188, 161, 160, 135, 124, 204,  40, 175,  80, 118, 113,  55]], dtype = np.uint8)
    
    encs = np.array([[229, 216, 143, 105,  68,  66, 237,  21, 238,  98, 122, 180, 118,  83, 184, 214],
                     [200,  90,  93,  86, 241, 222, 196, 116,  14, 187, 222,   8,   0,  26,   1, 143],
                     [206,   9, 185,  14, 134,  61, 173,  63, 127, 236, 244,  20, 161, 129, 122, 184],
                     [ 43,  32, 172, 124, 253,  64,  43,  27, 248, 153, 144, 115,  77,  23, 166,  34],
                     [239,  54, 237, 133, 251, 123, 148,  95, 143,  12,  67, 121, 148, 178,  41,  58],
                     [ 49, 129, 171, 190, 122,  68, 105, 243, 173, 196, 172,  65, 138, 204,  74, 177]], dtype = np.uint8)
    
    for i in range(6):
        value = np.random.rand(16)
        plain = plains[i,:]
        enc = encs[i,:]
        proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    return proj


@pytest.fixture
def base_class(base_project: Project):
    instance = AECPA(base_project, cwa.leakage_models.sbox_output, algorithm = cwma.attacks.LAE_CPA, 
                     epoch = 3, batch_size = 500) # type: ignore
    return instance

def test_algorithm(base_class: AECPA) -> None:
    assert type(base_class.algorithm) == cwma.attacks.LAE_CPA
    
    base_class.algorithm = CPAProgressive # type: ignore
    assert type(base_class.algorithm) == CPAProgressive

def test_point_range(base_class: AECPA) -> None:
    assert base_class.point_range == [0,16]
    
    with pytest.raises(TypeError):
        base_class.point_range = 10
        
    base_class.point_range = [0,10]
    assert base_class.point_range == [0,10]
    
def test_trace_range(base_class:AECPA) -> None:
    assert base_class.trace_range == [0,6]
    
    with pytest.raises(TypeError):
        base_class.trace_range = 10 # type: ignore
        
    base_class.trace_range = [0, 5]
    assert base_class.trace_range == [0, 5]

    
def test_epoch(base_class: AECPA) -> None:
    assert base_class.epoch== 3
    
    with pytest.raises(TypeError):
        base_class.epoch = [] # type: ignore
        
    base_class.epoch = 5
    assert base_class.epoch == 5  
    
def test_batch_size(base_class: AECPA) -> None:
    assert base_class.batch_size == 500
    
    with pytest.raises(TypeError):
        base_class.batch_size = [] # type: ignore
        
    base_class.batch_size = 100
    assert base_class.batch_size == 100

def test_group_num(base_class: AECPA) -> None:
    assert base_class.group_num == 1
    
    with pytest.raises(TypeError):
        base_class.group_num = [] # type: ignore
        
    base_class.group_num = 5
    assert base_class.group_num == 5
    
def test_convolve_vector(base_class: AECPA) -> None:
    assert base_class.convolve_vector == None
    
    with pytest.raises(TypeError):
        base_class.convolve_vector = 2 # type: ignore
        
    base_class.convolve_vector = np.arange(6) # type: ignore
    assert (base_class._convolve_vector == np.arange(6)).all()





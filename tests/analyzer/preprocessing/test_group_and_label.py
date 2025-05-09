import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from cw_ml_plugin.analyzer.preprocessing.convolve import Convolve
from chipwhisperer.common.api.ProjectFormat import Project

@pytest.fixture
def convolve_instance() -> Convolve:
    proj = Project()
    instance = Convolve(proj)
    return instance

def test_set_weight_vector(convolve_instance: Convolve) -> None:
    with pytest.raises(TypeError):
        convolve_instance.weight_vector = None
        
    convolve_instance.weight_vector = np.arange(5)
    assert (convolve_instance.weight_vector == np.arange(5)).all()
    
def test_convolve_mode(convolve_instance: Convolve) -> None:
    with pytest.raises(TypeError):
        convolve_instance.convolve_mode = [] # type: ignore
    assert convolve_instance.convolve_mode == 'same'

def test_convolve(convolve_instance: Convolve) -> None:
    input_array = np.ones(6)
    convolve_instance.weight_vector = np.arange(3)[::-1]+1
    output_array = convolve_instance._convolve(input_array)
    assert (output_array == np.array([5, 6, 6, 6, 6, 3])).all()

def test_project_convolve() -> None:
    proj = cw.create_project("test_proj")
    plain = enc = key = np.zeros(16, dtype = np.uint8)
    value = np.array([2,3,4,1,2,4], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    value = np.array([1,1,2,2,3,3], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    value = np.array([5,4,2,1,1,3], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    
    convolve_instance = Convolve(proj)
    convolve_instance.weight_vector = np.arange(3)[::-1]+1
    processed_proj = convolve_instance.preprocess()
    
    assert (processed_proj.waves[0] == np.array([13, 20, 14, 12, 17, 10])).all() # type: ignore
    assert (processed_proj.waves[1] == np.array([ 5,  9, 11, 15, 17,  9])).all() # type: ignore
    assert (processed_proj.waves[2] == np.array([22, 19, 11,  7, 12,  7])).all() # type: ignore
    
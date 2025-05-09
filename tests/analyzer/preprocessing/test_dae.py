import pytest
import numpy as np

import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from chipwhisperer.common.api.ProjectFormat import Project

from cw_ml_plugin.analyzer.preprocessing.dae import DAE

@pytest.fixture
def proj() -> Project:
    project = Project()
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.uint8)
    pt = np.array([101, 241, 251, 129,  57,  75,  32,  21,  21, 130,  75,  10,  42,60, 136,  30], dtype = np.uint8)
    et = np.array([ 89,  37, 156,  22,  86,  24,  60,  24, 223, 154, 165, 227, 212, 84, 208, 154], dtype = np.uint8)
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9, 207,  79,  60], dtype = np.uint8)
    project.traces.append(cw.Trace(value, pt, et, key)) # type: ignore
    
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.uint8)
    pt = np.array([ 75, 166,  12, 116, 200,   3,  98, 183,   5,   2, 221,  55, 223, 24, 253, 227], dtype = np.uint8)
    et = np.array([203, 173, 247,  83,  92,   6, 139,  18,  56, 212, 199, 130, 124, 30,  49,   1], dtype = np.uint8)
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9,   207,  79,  60], dtype = np.uint8)
    project.traces.append(cw.Trace(value, pt, et, key)) # type: ignore
    
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.uint8)
    pt = np.array([211, 232, 210,  42,  77, 108, 184, 115, 237,  22, 155, 135,   5, 151,  24,   4], dtype = np.uint8)
    et = np.array([ 80, 201,  84,   1, 109, 221, 241, 210,   9, 205,  69, 135, 199, 76, 142, 197], dtype = np.uint8)
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9, 207,  79,  60], dtype = np.uint8)
    project.traces.append(cw.Trace(value, pt, et, key)) # type: ignore
    
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.uint8)
    pt = np.array([105, 155, 121, 196, 211,  19, 132, 231, 239, 130,  79,  84, 203, 238,  87, 121], dtype = np.uint8)
    et = np.array([191, 102, 244, 117,  25,   7, 218, 164,  83, 137, 236, 203, 217, 32,  98, 113], dtype = np.uint8)
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9,   207,  79,  60], dtype = np.uint8)
    project.traces.append(cw.Trace(value, pt, et, key)) # type: ignore
    
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.uint8)
    pt = np.array([145, 157,  21, 192, 187, 174, 102,  37,  58, 227,  19, 208,   3, 23, 220,  77], dtype = np.uint8)
    et = np.array([ 19,  80, 215,  95, 239, 254,  36, 206, 146,  65, 188, 235, 249, 29,  81,  79], dtype = np.uint8)
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9,   207,  79,  60], dtype = np.uint8)
    project.traces.append(cw.Trace(value, pt, et, key)) # type: ignore
    
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.uint8)
    pt = np.array([113,   7, 156, 121, 178,  60,  30, 231, 194, 188, 159, 188,  43, 38,  33,  43], dtype = np.uint8)
    et = np.array([ 47,  70, 174,  54, 179, 127, 235,  11,  42,  38, 242,  76,  54, 81, 106, 169], dtype = np.uint8)
    key = np.array([ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9,   207,  79,  60], dtype = np.uint8)
    project.traces.append(cw.Trace(value, pt, et, key)) # type: ignore
    return project

@pytest.fixture
def dae(proj:Project) -> DAE:
    instance = DAE(project= proj, 
                   leak_model=cwa.leakage_models.last_round_state_diff, 
                   epoch=10, 
                   batch_size=10, 
                   )

    return instance
    
def test_point_range(dae:DAE):
    assert dae.parameters._point_range == [0, 16]
    
    with pytest.raises(TypeError):
        dae.point_range = 5
    
    dae.point_range = [0, 5]
    assert dae.parameters._point_range == [0, 5]

def test_trace_num(dae:DAE):
    print(dae.trace_num)
    assert dae.trace_num == 6
        
    dae.trace_range = [0, 3]
    assert dae.trace_num == 3

def test_trace_range(dae:DAE):
    assert dae.trace_range == [0, 6]
    
    with pytest.raises(TypeError):
        dae.trace_range = 5 # type: ignore
    
    dae.trace_range = [0, 3]
    assert dae.trace_range == [0,3]
    
def test_epoch(dae:DAE):
    assert dae.epoch == 10
    
    with pytest.raises(TypeError):
        dae.epoch = '10' # type: ignore
        
    dae.epoch = 5
    assert dae.epoch == 5
    
def test_batch_size(dae:DAE):
    assert dae.batch_size == 10
    
    with pytest.raises(TypeError):
        dae.batch_size = '10' # type: ignore
        
    dae.batch_size = 5
    assert dae.batch_size == 5
    
def test_group_num(dae:DAE):
    assert dae.group_num == 1
    
    with pytest.raises(TypeError):
        dae.group_num = '1' # type: ignore
        
    dae.group_num = 5
    assert dae.group_num == 5
    
def test_convolve_vector(dae:DAE):
    assert dae.convolve_vector == None
    
    with pytest.raises(TypeError):
        dae.convolve_vector = '1' # type: ignore
        
    dae.convolve_vector = np.array([1, 2, 3])
    assert dae.convolve_vector.all() == np.array([1, 2, 3]).all()
    
def test_noize_range(dae:DAE):
    assert dae.noize_range == None
    
    with pytest.raises(TypeError):
        dae.noize_range = '1' # type: ignore
        
    dae.noize_range = [1, 2]
    assert dae.noize_range == [1, 2]
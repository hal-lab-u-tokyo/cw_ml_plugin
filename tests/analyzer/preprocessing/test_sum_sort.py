import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from cw_ml_plugin.analyzer.preprocessing.sum_sort import Sum_Sort
from chipwhisperer.common.api.ProjectFormat import Project

@pytest.fixture
def sum_sort_instance() -> Sum_Sort:
    proj = Project()
    plain = enc = key = np.zeros(16, dtype = np.uint8)
    value = np.array([2,3,4,1,2,4], dtype = np.float64) # Sum is 16
    proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    value = np.array([5,4,2,1,2,3], dtype = np.float64) # Sum is 17
    proj.traces.append(cw.Trace(value, plain+1, enc, key)) # type: ignore
    value = np.array([1,1,2,2,3,3], dtype = np.float64) # Sum is 12
    proj.traces.append(cw.Trace(value, plain+2, enc, key)) # type: ignore
    sum_sort_instance = Sum_Sort(proj)
    return sum_sort_instance

def test_make_trace_matrix(sum_sort_instance: Sum_Sort) -> None:
    trace_matrix = sum_sort_instance._make_trace_matrix()
    correct_matrix = np.array([[2,3,4,1,2,4],[5,4,2,1,2,3],[1,1,2,2,3,3]], dtype = np.float64)
    assert (trace_matrix == correct_matrix).all()
    
def test_sort_arg(sum_sort_instance: Sum_Sort) -> None:
    arg = sum_sort_instance.sort()
    assert (arg == np.array([2,0,1])).all()

def test_get_trace(sum_sort_instance: Sum_Sort) -> None:
    processed_proj = sum_sort_instance.preprocess()
    
    assert (processed_proj.waves[0] == np.array([1,1,2,2,3,3])).all() # type: ignore
    assert (processed_proj.waves[1] == np.array([2,3,4,1,2,4])).all() # type: ignore
    assert (processed_proj.waves[2] == np.array([5,4,2,1,2,3])).all() # type: ignore
    assert (processed_proj.textins[0] == np.zeros(16, dtype = np.uint8)+2).all() # type: ignore
    assert (processed_proj.textins[1] == np.zeros(16, dtype = np.uint8)).all() # type: ignore
    assert (processed_proj.textins[2] == np.zeros(16, dtype = np.uint8)+1).all() # type: ignore
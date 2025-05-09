import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from cw_ml_plugin.analyzer.preprocessing.max_sort import Max_Sort
from chipwhisperer.common.api.ProjectFormat import Project

@pytest.fixture
def max_sort_instance() -> Max_Sort:
    proj = Project()
    plain = enc = key = np.zeros(16, dtype = np.uint8)
    value = np.array([2,3,4,1,2,4], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain+1, enc, key)) # type: ignore
    value = np.array([1,1,2,2,3,3], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain+2, enc, key)) # type: ignore
    value = np.array([5,4,2,1,1,3], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain+3, enc, key)) # type: ignore
    max_sort_instance = Max_Sort(proj)
    return max_sort_instance

def test_make_trace_matrix(max_sort_instance: Max_Sort) -> None:
    trace_matrix = max_sort_instance._make_trace_matrix()
    correct_matrix = np.array([[2,3,4,1,2,4],[1,1,2,2,3,3],[5,4,2,1,1,3]], dtype = np.float64)
    assert (trace_matrix == correct_matrix).all()
    
def test_sort_arg(max_sort_instance: Max_Sort) -> None:
    arg = max_sort_instance.sort()
    assert (arg == np.array([1,0,2])).all()

def test_get_trace(max_sort_instance: Max_Sort) -> None:
    processed_proj = max_sort_instance.preprocess()
    
    assert (processed_proj.waves[0] == np.array([1,1,2,2,3,3])).all() # type: ignore
    assert (processed_proj.waves[1] == np.array([2,3,4,1,2,4])).all() # type: ignore
    assert (processed_proj.waves[2] == np.array([5,4,2,1,1,3])).all() # type: ignore
    assert (processed_proj.textins[0] == np.zeros(16, dtype = np.uint8)+2).all() # type: ignore
    assert (processed_proj.textins[1] == np.zeros(16, dtype = np.uint8)+1).all() # type: ignore
    assert (processed_proj.textins[2] == np.zeros(16, dtype = np.uint8)+3).all() # type: ignore
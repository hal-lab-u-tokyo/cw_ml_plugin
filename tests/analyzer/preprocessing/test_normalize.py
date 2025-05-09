import numpy as np
import pytest
import chipwhisperer as cw
import cw_ml_plugin as cwma

from chipwhisperer.common.api.ProjectFormat import Project
from cw_ml_plugin.analyzer.preprocessing.normalize import Normalize

@pytest.fixture
def norm_trace() -> np.ndarray:
    norm_trace = np.array([[-0.14741036,  0.97609562, -0.85657371,  0.2749004 ,  0.55378486,  0.6812749 ,  0.29880478,  0.67330677,
                            -0.29083665, -0.03585657, -0.80079681, -0.96812749, -0.22709163,  0.83266932,  0.03585657, -0.41035857],
                            [-0.41832669,  0.30677291, -0.92031873, -0.09163347,  0.57768924,-0.99203187, -0.23505976,  0.44223108,
                            -0.97609562, -1        , 0.74501992, -0.57768924,  0.76095618, -0.8247012 ,  1        , 0.79282869]], dtype = np.float64)
    return norm_trace

@pytest.fixture
def proj() -> Project:
    proj = Project()
    value = np.array([109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76], dtype = np.float64)
    plain = enc = key = np.zeros(16, dtype= np.uint8)
    proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    value = np.array([ 75, 166,  12, 116, 200,   3,  98, 183,   5,   2, 221,  55, 223, 24, 253, 227], dtype = np.float64)
    proj.traces.append(cw.Trace(value, plain+1, enc+1, key+1)) # type: ignore
    return proj

def test_make_trace_matrix(proj:Project) -> None:
    norm = Normalize(proj)
    matrix = np.array([[109, 250,  20, 162, 197, 213, 165, 212, 91, 123,  27, 6, 99, 232, 132, 76],
                       [ 75, 166,  12, 116, 200,   3,  98, 183,   5,   2, 221,  55, 223, 24, 253, 227]], dtype = np.float64)
    assert (matrix == norm._make_trace_matrix()).all()

def test_norm(proj:Project, norm_trace:np.ndarray) -> None:
    norm = Normalize(proj)
    matrix = norm.norm(norm._make_trace_matrix())
    assert np.allclose(matrix, norm_trace)
    
def test_final(proj:Project, norm_trace:np.ndarray) -> None:
    norm = Normalize(proj)
    output_proj = norm.preprocess()
    for i in range(2):
        assert np.allclose(output_proj.waves[i], norm_trace[i])
        assert ((output_proj.textins[i] == np.zeros(16, dtype = np.uint8)+i)).all() # type: ignore
        assert ((output_proj.textouts[i] == np.zeros(16, dtype = np.uint8)+i)).all() # type: ignore
        assert ((output_proj.keys[i] == np.zeros(16, dtype = np.uint8)+i)).all() # type: ignore
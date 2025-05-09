import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from chipwhisperer.common.api.ProjectFormat import Project
from cw_ml_plugin.analyzer.preprocessing.poi_slice import Slice

@pytest.fixture
def project() -> Project:
    proj = Project()
    for i in range(6):
        value = plain = enc = key = np.ones(16, dtype = np.uint8) * (i + 1)
        proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    return proj

def test_range(project:Project) -> None:
    instance = Slice(project)
    with pytest.raises(TypeError):
        instance.poi = 1 # type: ignore
    instance.poi = [2, 5]
    assert instance.poi == [2, 5]

def test_trace_num(project:Project) -> None:
    instance = Slice(project)
    with pytest.raises(TypeError):
        instance.trace_num = 1 # type: ignore
    instance.trace_num = [3,5]
    assert instance.trace_num == [3,5]
    
def test_length_slice(project:Project) -> None:
    instance = Slice(project)
    instance.poi = [2, 5]
    slice_project = instance.preprocess()
    for n in range(6):
        assert (slice_project.waves[n] == project.waves[n][2:5]).all() # type: ignore
    
def test_trace_num_slice(project: Project) -> None:
    instance = Slice(project)
    instance.trace_num = [3,5]
    slice_project = instance.preprocess()
    assert  len(slice_project.traces) == 2
    for n in range(5-3):
        assert (slice_project.waves[n] == project.waves[n+3]).all() # type: ignore
    
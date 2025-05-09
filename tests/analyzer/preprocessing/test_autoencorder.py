import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from chipwhisperer.common.api.ProjectFormat import Project
from cw_ml_plugin.analyzer.preprocessing.autoencorder import Auto_Encorder

@pytest.fixture
def input_project() -> Project:
    proj = Project()
    for i in range(6):
        value = plain = enc = key = np.ones(16, dtype = np.uint8) * (i + 1)
        proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    return proj

@pytest.fixture
def target_project() -> Project:
    proj = Project()
    for i in range(6,12):
        value = plain = enc = key = np.ones(16, dtype = np.uint8) / (i + 1)
        proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
    return proj

def test_epoch(input_project: Project, target_project: Project) -> None:
    instance = Auto_Encorder(input_project, target_project)
    assert instance.epoch == None
    
    with pytest.raises(TypeError):
        instance.epoch = [] # type: ignore
        
    instance.epoch = 10
    assert instance.epoch == 10
    
def test_batch(input_project: Project, target_project: Project) -> None:
    instance = Auto_Encorder(input_project, target_project)
    assert instance.batch_size == 500
    
    with pytest.raises(TypeError):
        instance.batch_size = [] # type: ignore
        
    instance.batch_size = 10
    assert instance.batch_size == 10

def test_trace_length(input_project: Project, target_project: Project) -> None:
    proj = Project()
    plain = enc = key = np.ones(16, dtype = np.uint8)
    proj.traces.append(cw.Trace(np.ones(0), plain, enc, key)) # type: ignore
    
    with pytest.raises(ValueError):
        Auto_Encorder(proj, proj)
    
    instance = Auto_Encorder(input_project, target_project)
    assert instance.trace_length == 16
        
def test_trace_num(input_project: Project, target_project: Project) -> None:
    proj = Project()

    with pytest.raises(ValueError):
        Auto_Encorder(proj, proj)
        
    plain = enc = key = np.ones(16, dtype = np.uint8)
    input_project.traces.append(cw.Trace(np.ones(16), plain, enc, key)) # type: ignore
    with pytest.raises(ValueError):
        Auto_Encorder(input_project, target_project)
        
    target_project.traces.append(cw.Trace(np.ones(16), plain, enc, key)) # type: ignore
    instance = Auto_Encorder(input_project, target_project)
    assert instance.trace_num == 7
        
def test_name(input_project: Project, target_project:Project):
    instance = Auto_Encorder(input_project, target_project)
    assert instance.name == ""
    
    with pytest.raises(TypeError):
        instance.name = []
    
    instance.name = "test_proj,test_proj"
    assert instance.name == "test_proj,test_proj"
    
def test_make_trace(input_project: Project, target_project:Project):
    instance = Auto_Encorder(input_project, target_project)
    matrix = instance._make_trace_matrix(input_project)
    assert matrix.shape == (6, 16)
        
def test_run(input_project: Project, target_project:Project):
    instance = Auto_Encorder(input_project, target_project)
    instance.epoch = 3
    instance.batch_size = 2
    instance.name = "test_proj3"
    
    proj = instance.run()
    assert len(proj.traces) == len(input_project.traces) # type: ignore
    for i in range(len(proj.traces)): # type: ignore
        assert (proj.waves[i] != input_project.waves[i]).all() # type: ignore
        assert (proj.textins[i] == input_project.textins[i]).all() # type: ignore
        assert (proj.textouts[i] == input_project.textouts[i]).all() # type: ignore
        assert (proj.keys[i] == input_project.keys[i]).all() # type: ignore
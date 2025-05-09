import pytest
import numpy as np
from scipy.stats import norm
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from chipwhisperer.common.api.ProjectFormat import Project
from cw_ml_plugin.analyzer.preprocessing.addnoize import AddNoize

@pytest.fixture
def med():
    return 0

@pytest.fixture
def sigma():
    return 100

@pytest.fixture
def instance(med, sigma) -> AddNoize:
    proj = Project()
    plain = enc = key = np.zeros(16, dtype = np.uint8)
    np.random.seed(0)  # 再現性のために乱数シードを設定
    trace = np.random.normal(loc=med, scale=sigma, size=(100, 50))  # 平均0, 標準偏差100.0の正規分布データ
    for raw in range(0,len(trace)):
        proj.traces.append(cw.Trace(trace[raw], plain, enc, key)) # type: ignore
    noize_range = [10,40]
    
    instance = AddNoize(proj)
    instance.noize_range = noize_range
    return instance

def test_noize_range(instance:AddNoize) -> None:
    assert instance.noize_range == [10,40]
    
    with pytest.raises(TypeError):
        instance.noize_range = (1,1) # type: ignore
    instance.noize_range = [1,3]
    assert instance.noize_range == [1,3]

def test_make_trace_matrix(instance: AddNoize) -> None:
    trace_matrix = instance._make_trace_matrix()
    correct_matrix_shape = (100,50)
    assert trace_matrix.shape == correct_matrix_shape
    
def test_estimate_noize_model(instance: AddNoize, med, sigma) -> None:
    param, noizy_trace = instance._estimate_noize_model([10,40])
    # 結果を検証
    assert np.isclose(param[0], 0, atol=0.2), f"平均値が期待値から外れています: {param[0]}"
    assert np.isclose(param[1], sigma, atol=sigma*0.02), f"標準偏差が期待値から外れています: {param[1]}"
    
def test_preprocess(instance: AddNoize, med, sigma) -> None:
    trace = instance._make_trace_matrix()
    noizy = instance.preprocess()
    
    noizy_trace = np.empty((noizy._traceManager.num_traces(),noizy._traceManager.num_points()))
    for raw in range (0, noizy._traceManager.num_traces()):
        noizy_trace[raw] = noizy._traceManager.get_trace(raw) - trace[raw]
    
    param = norm.fit(noizy_trace)
    trace_height = noizy._traceManager.get_trace(0).max() - noizy._traceManager.get_trace(0).min()
    assert np.isclose(param[0], 0, atol=trace_height * 0.05), f"平均値が期待値から外れています: {param[0]}"
    assert np.isclose(param[1], sigma, atol=sigma*0.05), f"標準偏差が期待値から外れています: {param[1]}"

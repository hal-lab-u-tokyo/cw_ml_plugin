import pytest
import numpy as np
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from chipwhisperer.common.api.ProjectFormat import Project
from cw_ml_plugin.analyzer.attacks.LeakageAutoEncorder import LAE, onesubkey


@pytest.fixture
def project() -> Project:
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
def instance(project: Project) -> LAE:
    instance = LAE(project, cwa.leakage_models.sbox_output, 3, 500)
    return instance

def test_group_num(instance:LAE) -> None:
    assert instance.group_num == 1
    
    with pytest.raises(TypeError):
        instance.group_num = [] # type: ignore
        
    instance.group_num = 5
    assert instance.group_num == 5

def test_leakage_model(project:Project) -> None:
    with pytest.raises(TypeError):
        instance = LAE(project, []) # type: ignore
        
def test_point_range(instance:LAE) -> None:
    assert (instance.point_range == (0,16))
    
    with pytest.raises(TypeError):
        instance.point_range = []  # type: ignore
    
    instance.point_range = (2,5)
    assert (instance.point_range == (2,5))
    assert (instance.trace_length == 3) 

def test_trace_range(instance:LAE) -> None:
    assert (instance.trace_range == (0,6))
    
    with pytest.raises(TypeError):
        instance.trace_range = []  # type: ignore
        
    instance.trace_range = (0,5)
    assert (instance.trace_range == (0,5))
    assert instance.trace_num == 5
        
# def test_correlations(instance:LAE) -> None:
#     assert instance.correlations.shape == (16,256)

def test_update_interval(instance:LAE) -> None:
    assert instance.update_interval == 1000
    
    with pytest.raises(TypeError):
        instance.update_interval = [] # type: ignore
    
    instance.update_interval = 500
    assert instance.update_interval == 500
    
def test_convolve_vector(instance:LAE) -> None:
    assert instance.convolve_vector == None
    
    with pytest.raises(TypeError):
        instance.convolve_vector = [] # type: ignore
        
    instance.convolve_vector = np.arange(6)
    assert (instance._convolve_vector == np.arange(6)).all()
    
#Test for onesubkey
def test_calculate_slice(project:Project) -> None:
    instance = onesubkey(project, cwa.leakage_models.sbox_output)
    instance.calculate(tstart=2, tend=4, nbyte=0, nkey=0)
    assert (instance.trace == np.array([project._traceManager.get_trace(2), project._traceManager.get_trace(3)])).all()
    assert (instance.pt == np.array([[125,  83, 157, 235,  46,  82, 182,  53,  25,  57,  10,  40,  74, 211, 133, 102],
                                    [236, 192,  20, 106,  53, 193,  73, 142, 180, 165, 103, 161,  37, 70, 103, 253]], dtype=np.uint8)
            ).all()
    assert (instance.ct == np.array([[206,   9, 185,  14, 134,  61, 173,  63, 127, 236, 244,  20, 161, 129, 122, 184],
                                    [43,  32, 172, 124, 253,  64,  43,  27, 248, 153, 144, 115,  77,  23, 166,  34]], dtype=np.uint8)
            ).all()

def test_hyp(project: Project) -> None:
    instance = onesubkey(project, cwa.leakage_models.sbox_output)
    instance.calculate(tstart=2, tend=4, nbyte=0, nkey=0)
    for n in range(2,4):
        pt = project._traceManager.get_textin(n)
        ct = project._traceManager.get_textout(n)
        assert instance.hyp[n-2] == cwa.leakage_models.sbox_output.leakage(pt, ct, 0, 0, None)

def test_sum(project: Project) -> None:
    nbyte = 0
    nkey = 0
    instance = onesubkey(project, cwa.leakage_models.sbox_output)
    instance.calculate(tstart=2, tend=4, nbyte=nbyte, nkey=nkey)
    
    trace2 = project._traceManager.get_trace(2)
    trace3 = project._traceManager.get_trace(3)
    sum_trace = trace2 + trace3
    square_sum_trace = np.square(trace2) + np.square(trace3)
    assert np.allclose(instance.sum_trace, sum_trace)
    assert np.allclose(instance.square_sum_trace, square_sum_trace)
    
    hyp = [8,5]
    assert instance.sum_hyp[nkey] == 8 + 5
    assert instance.square_sum_hyp[nkey] == 64 + 25
    sum_ht = trace2 * 8 + trace3 * 5
    print('answer', sum_ht)
    print('cal', instance.sum_ht[nkey])
    assert np.allclose(instance.sum_ht[nkey], sum_ht)
    
def test_corr(project: Project) -> None:
    nbyte = 0
    nkey = 0
    
    instance = onesubkey(project, cwa.leakage_models.sbox_output)
    corr = instance.calculate(tstart=2, tend=6, nbyte=nbyte, nkey=nkey)
    
    hyp = np.empty(4, dtype = np.uint8)
    for n in range(2,6):
            pt = project._traceManager.get_textin(n)
            ct = project._traceManager.get_textout(n)
            hyp[n-2] = cwa.leakage_models.sbox_output.leakage(pt, ct, nkey, nbyte, None)
    trace = np.empty((4,16), dtype = np.float64)
    for i in range(4):
        trace[i] = project._traceManager.get_trace(i+2)

    answer = np.empty(16, dtype = np.longdouble)
    for i in range(16):
        answer[i] = np.corrcoef(hyp, trace[:,i])[0][1]
    
    assert np.allclose(corr, answer)
    
    
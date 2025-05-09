import pytest
import numpy as np
from scipy.spatial import distance
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from cw_ml_plugin.analyzer.preprocessing.hd_sort import HD_Sort
from chipwhisperer.common.api.ProjectFormat import Project

@pytest.fixture
def S_box() -> list:
    S_box = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    ]
    return S_box

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
def instance(proj:Project) -> HD_Sort:
    instance = HD_Sort(model = cwa.leakage_models.sbox_output, trace_source=proj, knum = 0, bnum= 0)
    return instance

def test_trace_length(instance: HD_Sort):
    assert instance.trace_length == 16
    
def test_trace_num(instance: HD_Sort):
    assert instance.trace_num == 6
    
def test_set_bnum(proj: Project) -> None:
    instance = HD_Sort(trace_source=proj, model = cwa.leakage_models.sbox_output, knum = 0, bnum= 0)        
    with pytest.raises(TypeError):
        instance.bnum = [] # type: ignore
        
    instance.bnum = 0
    assert instance.bnum == 0
    
def test_set_knum(proj: Project) -> None:
    instance = HD_Sort(trace_source=proj, model = cwa.leakage_models.sbox_output, knum = 0, bnum= 0)        
    with pytest.raises(TypeError):
        instance.knum = [] # type: ignore
        
    instance.knum = 0
    assert instance.knum == 0
    
def test_calculate_hd(proj: Project, S_box:list) -> None:
    instance = HD_Sort(model = cwa.leakage_models.sbox_output, trace_source=proj, knum = 0, bnum= 0)
    tnum = 0
    for bnum in range (10):
        knum = 0
        hd = instance.calculate_hd(tnum, bnum, knum)
        crypto = S_box[knum ^ proj.textins[0][bnum]] # type: ignore
        crypto_fill = bin(crypto)[2:].zfill(8)
        pre_crypto_fill = bin(0)[2:].zfill(8)
        correct_hd = distance.hamming(list(crypto_fill), list(pre_crypto_fill)) * 8
        assert hd == correct_hd
        
    for kcandidate in range (10):
        bnum = 0
        knum = kcandidate
        hd = instance.calculate_hd(tnum, bnum, knum)
        crypto = S_box[knum ^ proj.textins[0][bnum]] # type: ignore
        crypto_fill = bin(crypto)[2:].zfill(8)
        pre_crypto_fill = bin(0)[2:].zfill(8)
        correct_hd = distance.hamming(list(crypto_fill), list(pre_crypto_fill)) * 8
        assert hd == correct_hd
        
def test_sort(proj: Project, S_box:list) -> None:
    instance = HD_Sort(model = cwa.leakage_models.sbox_output, trace_source=proj, knum = 0, bnum= 0)
    instance.bnum = 0
    instance.knum = 0
    assert (np.array([4,0,5,2,1,3]) == instance.sort()).all()
    assert (np.array([2,4,5,6], dtype=np.uint8) == instance._HDs).all()
    assert (np.array([0, 1, 4, 5], dtype=np.uint8) == instance._HDs_index).all()
    
def test_final(proj:Project) -> None:
    instance = HD_Sort(model = cwa.leakage_models.sbox_output, trace_source=proj, knum = 0, bnum= 0)
    instance.bnum = 0
    instance.knum = 0
    preprocessed_proj = instance.preprocess()
    arg = [4,0,5,2,1,3]
    for i in range(6):
        assert (preprocessed_proj.textins[i] == proj.textins[arg[i]]).all() # type: ignore
        assert (preprocessed_proj.textouts[i] == proj.textouts[arg[i]]).all() # type: ignore        

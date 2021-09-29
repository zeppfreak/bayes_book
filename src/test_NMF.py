import pytest
import numpy as np
from NMF import NMFModel, init, update_S, update_T, update_V, sqsum

@pytest.fixture
def data():
    D, N = (3,4)
    X = np.ones(D*N).reshape(D, N)
    return X

@pytest.fixture
def model(data):
    D, N = data.shape
    K = 2
    a_t = 1.0
    b_t = 1.0
    a_v = 1.0
    b_v = 100.0
    model = NMFModel(a_t * np.ones((D,K)), b_t * np.ones((D,K)), a_v, b_v)
    return model

def test_sqsum():
    A = np.arange(3*4).reshape(3,4)
    print(f"A:\n {A}")
    print(A.shape)
    assert all([12,15,18,21] == sqsum(A, idx=0))
    assert all([6,22,38] == sqsum(A, idx=1))

def test_update_S(data, model):
    K = model.a_t.shape[1]
    D, N = data.shape
    S, A_t, B_t, A_v, B_v = init(data, model)
    S = update_S(data, A_t, B_t, A_v, B_v)
    assert (D,K,N) == S.shape
    print(f"S:\n {S}")

def test_update_T(data, model):
    K = model.a_t.shape[1]
    D, N = data.shape
    S, A_t, B_t, A_v, B_v = init(data, model)
    A_t, B_t = update_T(S, A_v, B_v, model)
    assert (D,K) == A_t.shape
    assert (D,K) == B_t.shape
    print(f"A_t:\n {A_t}")
    print(f"B_t:\n {B_t}")

def test_update_V(data, model):
    K = model.a_t.shape[1]
    D, N = data.shape
    S, A_t, B_t, A_v, B_v = init(data, model)
    A_v, B_v = update_V(S, A_t, B_t, model)
    assert (K,N) == A_v.shape
    assert (K,N) == B_v.shape
    print(f"A_v:\n {A_v}")
    print(f"B_v:\n {B_v}")
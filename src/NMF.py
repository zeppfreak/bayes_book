import numpy as np
from numpy.lib.shape_base import expand_dims
from scipy.special import psi, logsumexp

def sqsum(mt, idx):
    return np.sum(mt, axis=idx)

class NMFModel:
    def __init__(self, a_t, b_t, a_v, b_v):
        self.a_t = a_t
        self.b_t = b_t
        self.a_v = a_v
        self.b_v = b_v

def init(X, model):
    D, N = X.shape
    K = len(model.a_t[1])
    S = np.zeros((D, K, N))
    A_t = np.random.rand(D, K)
    B_t = np.random.rand(D, K)
    A_v = np.random.rand(K, N)
    B_v = np.random.rand(K, N)
    for d in range(D):
        for k in range(K):
            for n in range(N):
                S[d,k,n] = X[d,n] * A_t[d,k] * B_t[d,k] * A_v[k,n] * B_v[k,n]
    return S, A_t, B_t, A_v, B_v

def update_S(X, A_t, B_t, A_v, B_v):
    D, K = A_t.shape
    N = len(A_v[1])
    S = np.zeros((D, K, N))
    for d in range(D):
        for n in range(N):
            ln_P = (
                psi(A_t[d,:]) + np.log(B_t[d,:])
                + psi(A_v[:,n]) + np.log(B_v[:,n])
                )
            ln_P = ln_P - logsumexp(ln_P)
            S[d,:,n] = X[d,n] * np.exp(ln_P)
    
    return S

def update_T(S, A_v, B_v, model):
    D, K, N = S.shape
    a_t = model.a_t # D x K
    b_t = model.b_t # D x K
    A_t = a_t + sqsum(S, 2)
    B_t = 1.0/(a_t/b_t + np.tile(sqsum(A_v*B_v, idx=1), (D,1)))
    return A_t, B_t

def update_V(S, A_t, B_t, model):
    a_v = model.a_v # K x N
    b_v = model.b_v # K x N
    D, K, N = S.shape
    A_v = a_v + sqsum(S, 0)
    B_v = 1.0/(a_v/b_v + np.tile(sqsum(A_t*B_t, idx=0), (N,1)).T)
    return A_v, B_v

def update_model(A_t, B_t, model):
    return NMFModel(A_t, B_t, model.a_v, model.b_v)

def VI(X, model, max_iter):
    K = model.a_t[1]
    D, N = X.shape
    S, A_t, B_t, A_v, B_v = init(X, model)
    for iter in range(max_iter):
        S = update_S(X, A_t, B_t, A_v, B_v)
        A_v, B_v = update_V(S, A_t, B_t, model)
        A_t, B_t = update_T(S, A_v, B_v, model)
    
    return update_model(A_t, B_t, model), S, A_t*B_t, A_v*B_v
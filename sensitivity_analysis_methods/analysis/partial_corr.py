import numpy as np


def partial_corrcoef(X, y):
    '''
    computes the partial correlation coefficients of X and y
    returns the partial correlation coefficients
    '''
    newX = np.ones((X.shape[0], X.shape[1]+1))
    newX[:, :-1] = X
    correcoef = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_without_i = np.delete(newX, i, axis=1)
        lstsq_xi = np.linalg.lstsq(x_without_i, X[:, i], rcond=-1)[0]
        lstsq_yi = np.linalg.lstsq(x_without_i, y, rcond=-1)[0]
        part_X = X[:, i] - x_without_i.dot(lstsq_xi)
        part_Y = y - x_without_i.dot(lstsq_yi)
        corr = np.corrcoef(part_X, part_Y)
        correcoef[i] = corr[0][1]
    return correcoef

def generate_ranks(arr):
    '''
    computes the ranks of a array
    
    arr: np.array
    returns ranked array
    '''
    arr_ = arr.argsort()
    r_arr = np.empty_like(arr_)
    r_arr[arr_] = np.arange(len(arr))
    return r_arr+1
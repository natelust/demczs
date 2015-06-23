import numpy as np

def fit_func(p,t,ex):
    #return p[0]*np.sin((2*np.pi/p[1])*t - p[2]) + p[3]
    return p[0]*(t-p[1])**2+p[2]

def chi_func(model,data,errors,ex):
    return np.sum((model - data)**2/errors**2)

def con_func(par,bound):
    chi = 0
    for i,p in enumerate(par):
        if p>=bound[i][0] and p<=bound[i][1]:
            chi += 0
        else:
            chi += 1e10
    return chi

import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
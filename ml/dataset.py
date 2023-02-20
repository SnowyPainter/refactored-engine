import data
import numpy as np
import string
from sklearn.preprocessing import MinMaxScaler

MIN = 0
MAX = 2000 #-1 -- danger
scaler = MinMaxScaler()

#datasets, price.
cast = lambda v:float(str(v).translate(str.maketrans('', '', ',')))
Scale = lambda d:scaler.fit_transform(d.reshape(-1, 1))[:, 0]

def DateTerm(i):
    return data.Select(data.D(i).iloc[MIN:MAX], [data.DATE_TOKEN])[data.DATE_TOKEN]
def KOSPI200FIDX():
    return np.array(data.Select(data.D(1).iloc[MIN:MAX], [data.PRICE_TOKEN])[data.PRICE_TOKEN].apply(cast))
def NASDAQ100FIDX():
    return np.array(data.Select(data.D(2).iloc[MIN:MAX], [data.PRICE_TOKEN])[data.PRICE_TOKEN].apply(cast))
def SAMSUNGELECSTKP():
    samsung = data.Select(data.D(3).iloc[MIN:MAX], [data.PRICE_TOKEN])[data.PRICE_TOKEN].apply(cast)
    return np.array(samsung.apply(lambda p: p if(p) <= 1000000.0 else p/50.0))
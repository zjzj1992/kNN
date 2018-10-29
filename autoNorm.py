#data normalization

import numpy as np

def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVal,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet

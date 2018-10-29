#kNN model
import operator
import numpy as np

def classify(newData,dataSet,labels,k):
    #shape:dataSize
    #shape(data):Row number of data
    DataSetSize = np.shape(dataSet)[0]
    #tile data
    newData = np.tile(newData,(DataSetSize,1))
    #euclidean metric
    diffMat = newData - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #From small to large
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #k valuek
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    

#kNN model


import numpy as np


# create dataSet
def createData():
    dataSet = np.array([[5,5],
                  [4,6],
                  [3,3],
                  [7,4],
                  [4,4],
                  [2,2],
                  [8,8],
                  [10,10],
                  [11,11],
                  [7,7],
                  [7,8],
                  [5,5],
                  [6,2],
                  [6,6],
                  [9,9],
                  [3,5],
                  [5,2],
                  [1,1],
                  [9,6]])
    
    labels = ['正方形',
              '长方形',
              '正方形',
              '长方形',
              '正方形',
              '正方形',
              '正方形',
              '正方形',
              '正方形',
              '正方形',
              '长方形',
              '正方形',
              '长方形',
              '正方形',
              '正方形',
              '长方形',
              '长方形',
              '正方形',
              '长方形']
    return dataSet,labels

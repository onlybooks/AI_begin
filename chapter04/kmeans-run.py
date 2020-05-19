#-*- coding: utf-8 -*-
#

import sys
import numpy as np
from scipy.cluster.vq import kmeans, vq
from numpy import genfromtxt


def readData(filename, columns):
    csv = genfromtxt(filename, delimiter=",",
            usecols=(map(int, columns.split(","))))
    return csv


if __name__ == "__main__":

    cluster_filename = sys.argv[1]
    test_filename = sys.argv[2]
    columns = sys.argv[3]

    # 모델 읽어오기 (1)
    centroids = np.loadtxt(cluster_filename, delimiter=",")
    
    read_data = readData(test_filename, columns)

    # 결과 양자화 (2)
    idx, _ = vq(read_data, centroids)
    
    print(idx)

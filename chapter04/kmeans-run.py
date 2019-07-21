#-*- coding: utf-8 -*-
#

import sys
import numpy as np
from scipy.cluster.vq import kmeans,vq
from numpy import genfromtxt


# 테스트 데이터를 파일에서 읽어오는 함수
def readData(filename):
    csv = genfromtxt(filename, delimiter=',')
    return csv


if __name__ == "__main__":

    cluster_filename = sys.argv[1]
    test_filename = sys.argv[2]

    # 모델 읽어오기 (1)
    centroids = np.loadtxt(cluster_filename, delimiter=",")
    
    read_data = readData(test_filename)

    # 결과 양자화 (2)
    idx, _ = vq(read_data, centroids)
    
    print(idx)

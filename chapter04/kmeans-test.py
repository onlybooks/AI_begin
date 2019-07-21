#-*- coding: utf-8 -*-
#!/usr/bin/python

import sys
import csv
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from numpy import genfromtxt


def readData(filename):
    my_data = genfromtxt(filename, delimiter=',')
    return my_data


if __name__ == "__main__":

    k = sys.argv[1]
    data_filename = sys.argv[2]
    
    # 데이터 파일 읽기 (2)
    data = readData(data_filename)
    
    # 입력된 k 갯수 만큼 k-means 계산 (2)
    centroids, _ = kmeans(data, int(k))
    
	# 각 예제를 양자화를 통해 군집에 할당 (3)
    idx,_ = vq(data, centroids)

    # 결과를 출력 (4)
    for i in range(int(k)):
        plot(data[idx==i, 0], data[idx==i, 1], 'o',  markersize=3)
        plot(centroids[i:,0], centroids[i:,1], '^r', markersize=10)

    show()


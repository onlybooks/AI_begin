# -*- coding: utf-8 -*-

#pandas 모듈 추가
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 데이타 셋 로드 (1)
mushroom = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
        header = None)


# 문자를 수치로 치환 (2)
X = []
y = []

for row_index, row in mushroom.iterrows():
    y.append(row.ix[0])
    row_X = []
    for v in row.ix[1:]:
        row_X.append(ord(v)) # (3)
    X.append(row_X)
    

# 학습 데이터와 테스트 데이터 나누기 (4)
x_train, x_test, y_train, y_test = train_test_split(X, y)


# 학습 및 테스트 (5)
svc = SVC()
svc.fit(x_train, y_train)

print('훈련 점수 : {: .3f}'.format(svc.score (x_train, y_train)))
print('테스트 점수 : {: .3f}'.format(svc.score (X_test, y_test)))
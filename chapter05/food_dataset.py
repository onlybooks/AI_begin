#-*- coding: utf-8 -*-

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np

## 분류 데이터 로딩

root_dir = "./food/"
# 카테고리 명령 지정 (1)
categories = ["Chicken", "Dolsotbab", "Jeyugbokkeum", "Kimchi", 
                "Samgyeobsal", "SoybeanPasteStew"]
nb_classes = len(categories)

# 이미지 크기 지정 (2)
image_width = 64
image_height = 64


## 데이터 변수
X = [] # 이미지 데이터
Y = [] # 레이블 데이터

for idx, category in enumerate(categories):
    image_dir = root_dir + category
    files = glob.glob(image_dir + "/" + "*.jpg")
    print(image_dir + "/" +"*.jpg")

    for i, f in enumerate(files):
        # 이미지 로딩 (3)
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)

X = np.array(X)
Y = np.array(Y)

# 학습 데이터와 테스트 데이터 나누기 (4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

xy = (X_train, X_test, Y_train, Y_test)
# 데이터 파일 저장 (5)
np.save(root_dir + "kfood.npy", xy)

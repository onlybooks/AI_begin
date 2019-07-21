#-*- coding: utf-8 -*-

from keras.preprocessing.text import Tokenizer

def read_data(filename):
    with open(filename, 'r', encoding='utf8') as f:
        result = [line.split('\t') for line in f.read().splitlines()]
        result = result[1:] #header 제외
    return result

# 학습에 활용할 텍스트 파일 읽기
train_tmp = read_data('./ratings_train.txt')
test_tmp = read_data('./ratings_test.txt')

def kor_movie(max_num_words=1000):
    # 데이터 구조 : ID | 리뷰데이터 | 감성라벨
    # 필요없는 id를 제외하고 리뷰데이터과 감성라벨로만 이루어진 데이터를 만든다.
    # 학습용
    train_x = []
    train_y = []
    for i in range(len(train_tmp)):
       train_x.append(train_tmp[i][1])
       train_y.append(int(train_tmp[i][2]))

    # 테스트용
    test_x = []
    test_y = []
    for i in range(len(test_tmp)):
       test_x.append(test_tmp[i][1])
       test_y.append(int(test_tmp[i][2]))

    # 단어사전을 만들고, 문장을 단어사전에 맞게 자연수로 변형한다. (1)
    # 빈도수가 높은 단어순으로 max_num_words 개의 단어가 들어있는 사전생성
    tokenizer = Tokenizer(num_words=max_num_words) 
    tokenizer.fit_on_texts(train_x)

    # 위에서 만든 단어 사전을 기준으로 텍스트 데이터를 자연수 수열로 변환한다. (2)
    token_train_x = tokenizer.texts_to_sequences(train_x)
    token_test_x = tokenizer.texts_to_sequences(test_x)
    
    return (token_train_x, train_y) , (token_test_x, test_y)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

print('...전처리하기...')
max_num_words = 5000 # 단어사전 크기
maxlen = 100 # 문장의 최대길이
batch_size = 32 # 배치 사이즈

(x_train, y_train), (x_test, y_test) = kor_movie(max_num_words) # (3)

print(x_train[0])
print(y_train[0])

# 각 영화 리뷰 문장들의 길이가 다르다. (4)
# 딥러닝을 하기 위해서는 길이를 통일할 필요가 있다.
# 100 길이 안에서 데이터를 채우고, 길이가 부족해서,
# 빈 영역이 생기는 경우 0 으로 채운다.
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('...모델만들기...') # (5)
model = Sequential()
model.add(Embedding(max_num_words, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('...학습...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
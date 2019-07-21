#-*- coding: utf-8 -*-
# gensim 모듈 추가 (1)
from gensim import corpora 
from gensim import models

documents=[
'나는 아침에 라면을 자주 먹는다.',
'나는 아침에 밥 대신에 라면을 자주 먹는다.',
'현대인의 삶에서 스마트폰은 필수품이 되었다.',
'현대인들 중에서 스마트폰을 사용하지 않는 사람은 거의 없다. ',
'점심시간에 스마트폰을 이용해 영어 회화 공부를 하느라 혼자 밥을 먹는다.'
]

# 불용어 제거(2) 
stoplist = ('.!?')
texts = [[word for word in document.split() if word not in stoplist]
        for document in documents]

# 사전과 코퍼스 만들기 (3) 
# 사전만들기.
dictionary = corpora.Dictionary(texts)

# 코퍼스 만들기(벡터화)
corpus = [dictionary.doc2bow(text) for text in texts]
print ('corpus : {}'.format(corpus))

# 모델구축(4) 
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, 
        num_topics=2, random_state = 1)


# 주제마다 출현 확률이 높은 단어 순으로 출력 (5)
for t in lda.show_topics():
	print(t)


# 주제를 워드 크라우드로 시각화 하기(6)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 윈도우 OS의 폰트 경로 예시
# font_path = 'C:/Windows/Fonts/malgun.ttf';
# 우분투 OS의 폰트 경로 예시
# font_path='/Library/Fonts/AppleGothic.ttf'

wc = WordCloud(background_color='white',
        font_path='C:/Windows/Fonts/malgun.ttf')

plt.figure(figsize=(30,30))
for t in range(lda.num_topics):
    plt.subplot(5,4,t+1)
    x = dict(lda.show_topic(t,200))
    im = wc.generate_from_frequencies(x)
    plt.imshow(im)
    plt.axis("off")
    plt.title("Topic #" + str(t))

# 이미지 저장(7)
plt.savefig('LDA_wordcloud.png', bbox_inches='tight')

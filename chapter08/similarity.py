# -*- coding: utf-8 -*-

import sys
from gensim.models import word2vec # (1)

# 첫 번째 파라미터는 사용할 모델 파일 이름 (2)
model   = word2vec.Word2Vec.load(sys.argv[1])
results = model.most_similar(positive=['woman','king'],
		negative=['man'], topn=1) # (3)

for result in results: # (4)
    print(result[0], '\t', result[1])

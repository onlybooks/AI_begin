# -*- coding: utf-8 -*-

from konlpy.tag import Kkma # (1)


# (2)
kkma = Kkma()
# 문장 분리
print('kkma 문장분리 : ', 
		kkma.sentences(u'안녕하세요. 반갑습니다. 저는 인공지능입니다.'))   
# 명사 추출
print('kkma 명사만 추출 : ', 
		kkma.nouns(u'을지로 3가역 주변 첨단빌딩숲 사이에 자리 잡은 커피집'))
print('='*80)


from konlpy.tag import Twitter # (3)
# (4)
tagger = Twitter()
print('Twitter 명사만 추출 : ', 
		tagger.nouns(u'을지로 3가역 주변 첨단빌딩숲 사이에 자리 잡은 커피집'))
# 품사 태깅(Part-of-speech tagging)
print('Twitter 품사 추출 : ',
		tagger.pos(u'이것도 되나요ㅋㅋ'))
# 명사 원형 추출
print('Twitter 오타와 원형처리 : ',
		tagger.pos(u'이것도 되나요ㅋㅋ', norm=True, stem=True))
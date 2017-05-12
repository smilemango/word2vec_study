import multiprocessing
import sqlite3
import logging
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
conn = sqlite3.connect('movies.sqlite')
c = conn.cursor()

# SQL 쿼리 실행
c.execute("""
select movie_id, comment_id, rate, replace(comment, ',',' ')
from comments
where
     replace(comment, ',',' ') like '%전지현 %'
and  replace(comment, ',',' ') like '%하정우 %'
and  replace(comment, ',',' ') like '%이정재 %'
union all
select * from comments
where
    comment like '%전지현 %'
and comment like '%하정우 %'
and comment like '%조진웅 %'
""")

# 데이타 Fetch
rows = c.fetchall()

print("1.ROWS: %d" % len(rows))
rows = rows * 5
print("2.ROWS: %d" % len(rows))


docs_ko = []
for a_row in rows:
    words = a_row[3]
    docs_ko.append(words)

# Tokenize
from konlpy.tag import Twitter

t = Twitter()

#pos = lambda d: ['/'.join(p) for p in t.pos(d)]

def pos( d ):
    ret = []

    for p in t.pos(d):
        if p[1] == 'Josa' or p[1] =='Punctuation' or p[1] == 'Suffix' or p[1] == 'Eomi':
            continue
        else:
            ret.append( '/'.join( p ) )

    return ret

text_ko = [pos(doc) for doc in docs_ko]

#print(text_ko)

# train
from gensim.models import word2vec


wv_model_ko = word2vec.Word2Vec(
    window=10, min_count=11, size=2
)
wv_model_ko.build_vocab(text_ko)
print("Word2Vec vocabulary length:", len(wv_model_ko.wv.vocab))

wv_model_ko.train(text_ko,total_examples=wv_model_ko.corpus_count, epochs=wv_model_ko.iter)

#wv_model_ko.init_sims(replace=True)

wv_model_ko.save('twitter\\ko_word2vec_e.model')

# test
print(wv_model_ko.most_similar(pos('하정우')))
print(wv_model_ko.most_similar(pos('전지현')))
print(wv_model_ko.most_similar(pos('이정재')))

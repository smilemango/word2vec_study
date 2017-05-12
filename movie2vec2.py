import sqlite3
import gensim
import codecs
import glob
import logging
import multiprocessing
import os
import pprint as pp
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



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

sentences = []

idx = 0
for a_row in rows:
    words = a_row[3]
    sentences.append(words)

print("SENTENSE SIZE : %d " %  len(sentences) )
sentences = sentences * 20
print("SENTENSE SIZE : %d " %  len(sentences) )



# Tokenize
from konlpy.tag import Twitter

t = Twitter()
def pos( d ):
    ret = []
    for p in t.pos(d):
        if p[1] == 'Josa' or p[1] =='Punctuation' or p[1] == 'Suffix' or p[1] == 'Eomi':
            continue
        else:
            ret.append( '/'.join( p ) )
    return ret

text_ko = [pos(doc) for doc in sentences]

print(text_ko)

del sentences
#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


model = gensim.models.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


model.build_vocab(text_ko)
print("Word2Vec vocabulary length:", len(model.wv.vocab))

model.train(text_ko,total_examples=model.corpus_count, epochs=model.iter)

#my video - how to visualize a dataset easily
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = model.wv.syn0
#pp.pprint(model.wv.syn0[0])
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model.wv.vocab[word].index])
            for word in model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


pp.pprint(points.head(10))
sns.set_context("poster")

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import pylab

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


ax = points.plot.scatter("x", "y", s=10, figsize=(20, 12))
for i, point in points.iterrows():
    ax.text(point.x + 0.001, point.y + 0, point.word, fontsize=11)

print(model.most_similar(pos('하정우')[0]))
print(model.most_similar(pos('이정재')[0]))
print(model.most_similar(pos('전지현')[0]))


for value in points.values:
    if value[0] ==pos('하정우')[0]:
        print(value)


plt.show()
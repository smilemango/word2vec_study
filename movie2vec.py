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
    words = a_row[3].split()
    sentences.append(words)

print("SENTENSE SIZE : %d " %  len(sentences) )
sentences = sentences * 5
print("SENTENSE SIZE : %d " %  len(sentences) )

model = gensim.models.Word2Vec(window=10, min_count=11, size=2)
model.build_vocab(sentences)
print("Word2Vec vocabulary length:", len(model.wv.vocab))

model.train(sentences,total_examples=model.corpus_count, epochs=model.iter)

#my video - how to visualize a dataset easily
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = model.wv.syn0
pp.pprint(model.wv.syn0[0])
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
pp.pprint(model.most_similar('하정우'))


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

for value in points.values:
    if value[0] =='하정우':
        print(value)

plt.show()
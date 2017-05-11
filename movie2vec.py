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
c.execute("select * from comments order by comment_id limit 10 ")

# 데이타 Fetch
rows = c.fetchall()

sentences = []

idx = 0
for a_row in rows:
    words = a_row[3].split()
    sentences.append(words)

print("SENTENSE SIZE : %d " %  len(sentences) )

model = gensim.models.Word2Vec(window=10, min_count=1, size=3)
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


pp.pprint( points.head(10))

sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


plot_region(x_bounds=(3.0, 4.5), y_bounds=(-0.5, 0.0))

plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))
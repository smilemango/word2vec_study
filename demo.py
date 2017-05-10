#future is the missing compatibility layer between Python 2 and Python 3.
#It allows you to use a single, clean Python 3.x-compatible codebase to
#support both Python 2 and Python 3 with minimal overhead.
from __future__ import absolute_import, division, print_function

#encoding. word encodig
import codecs
#finds all pathnames matching a pattern, like regex
import glob
#log events for libraries
import logging
#concurrency
import multiprocessing
#dealing with operating system , like reading file
import os
#pretty print, human readable
import pprint
#regular expressions
import re

#natural language toolkit
import nltk
#word 2 vec
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#parse dataset
import pandas as pd
#visualization
import seaborn as sns


if not os.path.exists("trained/thrones2vec.w2v"):

    #stopwords like the at a an, unnecesasry
    #tokenization into sentences, punkt
    #http://www.nltk.org/
    nltk.download("punkt")
    nltk.download("stopwords")

    #get the book names, matching txt file
    book_filenames = sorted(glob.glob("data/*.txt"))

    #print books
    print("Found books:")
    book_filenames = [book_filenames[0]]
    print(book_filenames)

    #step 1 process data

    #initialize rawunicode , we'll add all text to this one bigass file in memory
    corpus_raw = u""
    #for each book, read it, open it un utf 8 format,
    #add it to the raw corpus
    for book_filename in book_filenames:
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))
        print()


    #tokenizastion! saved the trained model here
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #tokenize into sentences
    raw_sentences = tokenizer.tokenize(corpus_raw)

    #convert into list of words
    #remove unecessary characters, split into words, no hyhens and shit
    #split into words
    def sentence_to_wordlist(raw):
        clean = re.sub("[^a-zA-Z]"," ", raw)
        words = clean.split()
        return words

    #for each sentece, sentences where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))

    #print an example
    print(raw_sentences[5])
    print(sentence_to_wordlist(raw_sentences[5]))

    #count tokens, each one being a sentence
    token_count = sum([len(sentence) for sentence in sentences])
    print("The book corpus contains {0:,} tokens".format(token_count))

    #step 2 build our model, another one is Glove
    #define hyperparameters

    # Dimensionality of the resulting word vectors.
    #more dimensions mean more traiig them, but more generalized
    num_features = 300

    #
    # Minimum word count threshold.
    min_word_count = 3

    # Number of threads to run in parallel.
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 7

    # Downsample setting for frequent words.
    #rate 0 and 1e-5
    #how often to use
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    seed = 1

    thrones2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    thrones2vec.build_vocab(sentences)

    print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))

    #train model on sentneces
    thrones2vec.train(sentences,total_examples=thrones2vec.corpus_count, epochs=thrones2vec.iter)

    #save model
    if not os.path.exists("trained"):
        os.makedirs("trained")

    thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))

#load model
thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))

#squash dimensionality to 2
#https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

#put it all into a giant matrix
all_word_vectors_matrix = thrones2vec.wv.syn0

#train t sne
print("#train t sne")
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#plot point in 2d space
print("#plot point in 2d space")
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index]) for word in thrones2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)

#plot
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


plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))

plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))

thrones2vec.most_similar("Stark")

thrones2vec.most_similar("Aerys")

thrones2vec.most_similar("direwolf")

#distance, similarity, and ranking
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")


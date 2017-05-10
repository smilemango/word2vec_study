from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# load
from konlpy.corpus import kobill

docs_ko = [kobill.open(i).read() for i in kobill.fileids()]

# Tokenize
from konlpy.tag import Twitter;

t = Twitter()
pos = lambda d: ['/'.join(p) for p in t.pos(d)]
texts_ko = [pos(doc) for doc in docs_ko]

# train
from gensim.models import word2vec

wv_model_ko = word2vec.Word2Vec(texts_ko)
wv_model_ko.init_sims(replace=True)

wv_model_ko.save('ko_word2vec_e.model')

# test
print(wv_model_ko.most_similar(pos('소말리아')))

from gensim.models import word2vec
import logging

PATH2SOGOUT = ''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence(PATH2SOGOUT)
model = word2vec.Word2Vec(sentences, workers=20, sg=1, size=300, min_count=50)
model.wv.save_word2vec_format('300Tvectors.txt', binary=False)


from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import pickle

lda = LdaModel.load('./lda_model.gensim')

print(lda.print_topics(num_words=5))



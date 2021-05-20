from eunjeon import Mecab
from tqdm import tqdm 
import re
from gensim.models.ldamodel import LdaModel 
from gensim.models.callbacks import CoherenceMetric 
from gensim import corpora 
from gensim.models.callbacks import PerplexityMetric 
import logging 
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models.coherencemodel import CoherenceModel 
import matplotlib.pyplot as plt
from cv import compute_coherence_values
from cv import find_optimal_number_of_topics

### cv를 통해 찾은 parameter를 적용한 모델로 분석


if __name__=="__main__":
    #전처리
    processed_data = [sent.strip().split(",") for sent in tqdm(open("tokenized_data.csv",'r',encoding='utf-8').readlines())]
    dictionary = corpora.Dictionary(processed_data)
    dictionary.filter_extremes(no_below=10, no_above=0.05) 
    corpus = [dictionary.doc2bow(text) for text in processed_data] 
    print('Number of unique tokens: %d' % len(dictionary)) 
    print('Number of documents: %d' % len(corpus))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #모델 생성
    #u_mass: 정확도를 측정하는 지표
    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell') 
    coherence_logger = CoherenceMetric(corpus=corpus, coherence="u_mass", logger='shell') 
    #토픽 개수 10개
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=15, passes=30, 
    callbacks=[coherence_logger, perplexity_logger]) 
    #단어 5개만 출력
    topics = lda_model.print_topics(num_words=5) 
    #토픽 출력
    for topic in topics: 
        print(topic)

    #coherence score 측정(c_v)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='c_v') 
    coherence_lda = coherence_model_lda.get_coherence() 
    print('\nCoherence Score (c_v): ', coherence_lda)
    #coherence score 측정(u_mass)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence="u_mass") 
    coherence_lda = coherence_model_lda.get_coherence() 
    print('\nCoherence Score (u_mass): ', coherence_lda)

    #모델 저장
    pickle.dump(corpus, open('./lda_corpus.pkl', 'wb')) 
    dictionary.save('./lda_dictionary.gensim') 
    lda_model.save('./lda_model.gensim') 

    #시각화
    pyLDAvis.enable_notebook()
    pyLDAvis.gensim_models.prepare(lda_model,corpus,dictionary)







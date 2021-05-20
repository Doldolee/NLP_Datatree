from eunjeon import Mecab
from tqdm import tqdm
import re
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
import logging
import pickle
import pyLDAvis.gensim_models as gensim
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# 최적의 토픽 수 찾기(cross-validation)

# 모델 생성 및 모델 별 coherence점수 측정


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=6):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary,
                         num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# 최적의 토픽수 탐색


def find_optimal_number_of_topics(dictionary, corpus, processed_data):
    limit = 40
    start = 2
    step = 6

    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary, corpus=corpus, texts=processed_data, limit=limit)

    # 시각화그래프
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


# 실행
if __name__ == '__main__':
    processed_data = [sent.strip().split(",")for sent in tqdm(
        open('tokenized_data.csv', 'r', encoding='utf-8').readlines())]
    #정수 인코딩과 빈도수 생성
    dictionary = corpora.Dictionary(processed_data)
    # 짧은 단어 제거
    dictionary.filter_extremes(no_below=10, no_above=0.05)
    # corpus생성
    corpus = [dictionary.doc2bow(text) for text in processed_data]
    #clustering된 결과 출력
    # logging.basicConfig(
    #     format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    find_optimal_number_of_topics(dictionary, corpus, processed_data)

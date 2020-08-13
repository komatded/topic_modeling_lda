from gensim.models import CoherenceModel
from lda_model import LDAModel
from gensim import corpora
from utils_lda import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

text_len_threshold = 100
n_topics = np.arange(1, 30, 2)


def evaluate_model(data: pd.DataFrame, num_topics: int, text_data: list, dictionary: corpora.Dictionary, corpus: list):
    lda_model = LDAModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15,
                         start_date=data.published.min(), end_date=data.published.max())
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text_data, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    return perplexity, coherence


if __name__ == '__main__':
    scores = list()
    data = load_data('../resources/post.csv', text_len_threshold=text_len_threshold)
    weeks = split_weeks(df=data, day_shift=3.5)
    bar = tqdm(weeks, leave=True)
    for week in bar:
        perplexity_scores, coherence_scores = list(), list()
        text_data = week.lemmas.to_list()
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]
        for n in n_topics:
            bar.set_description("Num topics: {0}".format(n))
            bar.refresh()
            p, c = evaluate_model(data=week, num_topics=n, text_data=text_data, dictionary=dictionary, corpus=corpus)
            perplexity_scores.append(p)
            coherence_scores.append(c)
        scores.append({'perplexity_scores': perplexity_scores, 'coherence_scores': coherence_scores})
        pickle.dump(scores, open('../resources/evaluation_scores.pickle', 'wb'))

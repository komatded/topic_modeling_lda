import gensim
import numpy as np
from utils_lda import *


class Theme:
    def __init__(self):
        self.topics = list()
        self.data = list()

    def add_topic(self, topic, weight):
        self.topics.append(topic)
        topic_data = {'start_date': topic.start_date, 'end_date': topic.end_date, 'junction_weight': weight}
        topic_data.update(topic.key_words)
        self.data.append(topic_data)

    def get_table(self, full=True):
        df = pd.DataFrame(self.data).fillna(0)
        if not full:
            return df.loc[:, (df != 0).all(axis=0)]
        return df

    def save(self, fp):
        df = pd.DataFrame(self.data).fillna(0).round(5)
        df.to_csv(fp, index=False)


class Topic:
    def __init__(self, lda_model, topic_id: int):
        self.id = topic_id
        self.start_date = lda_model.start_date
        self.end_date = lda_model.end_date
        self.string = lda_model.print_topic(self.id)
        self.vec = lda_model.get_topics()[self.id]
        self.key_words = {lda_model.id2word[word_id]: weight for word_id, weight in lda_model.get_topic_terms(self.id)}

    def __repr__(self):
        return '<Topic id:{0}, start_date:{1}, end_date:{2}>'.format(self.id, self.start_date, self.end_date)

    def __eq__(self, other):
        if isinstance(other, Topic):
            eq_id = self.id == other.id
            eq_dates = self.start_date == other.start_date and self.end_date == other.end_date
            eq_vec = np.array_equal(self.vec, other.vec)
            return eq_id and eq_dates and eq_vec
        return False


class LDAModel(gensim.models.ldamodel.LdaModel):
    def __init__(self, start_date, end_date, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word2id = self.id2word.token2id
        self.start_date = start_date
        self.end_date = end_date

    def predict_raw_document_topic(self, raw_text, min_word_len=4, exclude_hashtags=True, lemmatize=True):
        words = preprocess_text(raw_text, min_word_len, exclude_hashtags, lemmatize)
        return self.predict_preprocessed_document(words)

    def predict_preprocessed_document(self, words):
        bow = self.id2word.doc2bow(words)
        document_topics = self.get_document_topics(bow=bow)
        return sorted(document_topics, key=lambda i: i[1])[::-1]

    def get_similar_topics(self, lda_model, distance_func):
        same_words = list(set(self.id2word.values()) & set(lda_model.id2word.values()))
        topics_1 = self.get_topics()
        topics_2 = lda_model.get_topics()
        ind_1 = np.array([self.word2id[word] for word in same_words])
        ind_2 = np.array([lda_model.word2id[word] for word in same_words])
        topics_1_cut = np.array([topic[ind_1] for topic in topics_1])
        topics_2_cut = np.array([topic[ind_2] for topic in topics_2])
        mdiff = np.zeros((topics_2.shape[0], topics_1.shape[0]))
        for i2, i1 in np.ndindex(mdiff.shape):
            mdiff[i2][i1] = distance_func(topics_1_cut[i1], topics_2_cut[i2])
        return mdiff

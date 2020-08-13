from utils_lda import *
from lda_model import LDAModel
from gensim.corpora import Dictionary

start_date = datetime.datetime(2020, 5, 1)
end_date = start_date + datetime.timedelta(days=7)
week_aud = load_data('../resources/post_children_aud.csv', text_len_threshold=100,
                     start_date=start_date, end_date=end_date)
week_hashtags = load_data('../resources/post_children_hashtags.csv', text_len_threshold=100,
                          start_date=start_date, end_date=end_date)


def train_model(data: pd.DataFrame, num_topics: int):
    text_data = data.lemmas.to_list()
    dictionary = Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    lda_model = LDAModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15,
                         start_date=data.published.min(), end_date=data.published.max())
    return lda_model, corpus


# model_hashtags_1w, corpus_hashtags_1w = train_model(data=week_hashtags, num_topics=10)
# model_aud_1w, corpus_aud_1w = train_model(data=week_aud, num_topics=10)

# if __name__ == '__main__':

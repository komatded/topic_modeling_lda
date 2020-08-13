from lda_model import LDAModel
from utils_lda import preprocess_text
from gensim.corpora import Dictionary


def main(texts: list, num_topics=10):
    topics = list()
    lemmatized = [preprocess_text(raw_text=text, min_word_len=4, exclude_hashtags=False, lemmatize=True)
                  for text in texts]
    dictionary = Dictionary(lemmatized)
    corpus = [dictionary.doc2bow(text) for text in lemmatized]
    lda_model = LDAModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15,
                         start_date=None, end_date=None)
    for i in range(num_topics):
        words, weights = list(), list()
        terms = lda_model.get_topic_terms(i, topn=10)
        for word_id, weight in terms:
            words.append(lda_model.id2word[word_id])
            weights.append(weight)
        topics.append({'id': i, 'key_words': words, 'weights': weights})
    return topics

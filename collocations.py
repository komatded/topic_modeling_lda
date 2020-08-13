from gensim.models import Phrases


def extract_collocations(documents_lemmas, min_count, threshold):
    bigram = Phrases(documents_lemmas, min_count=min_count, threshold=threshold)
    return list(bigram.export_phrases(documents_lemmas))

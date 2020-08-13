from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import pickle


eval_results = pickle.load(open('../resources/evaluation_scores.pickle', 'rb'))
coherence = np.array([i['coherence_scores'] for i in eval_results])
perplexity = np.array([i['perplexity_scores'] for i in eval_results])

n_topics = np.arange(1, 30, 2)
weeks_len = np.array([80943, 91451, 127936, 138946, 203835, 148427, 44382, 39641, 37447, 41271, 40199, 37699])

data_coh = defaultdict(list)
data_per = defaultdict(list)

for n in range(len(weeks_len)):
    for m in range(len(n_topics)):
        data_coh[(weeks_len[n] / n_topics[m]) // 10000].append(coherence[n][m])
        data_per[(weeks_len[n] / n_topics[m]) // 10000].append(perplexity[n][m])

data_coh = {i: np.array(data_coh[i]).mean() for i in data_coh}
data_per = {i: np.array(data_per[i]).mean() for i in data_per}
x_coh = sorted([i for i in data_coh])
y_coh = [data_coh[i] for i in x_coh]
x_per = sorted([i for i in data_per])
y_per = [data_per[i] for i in x_per]

plt.plot(x_coh, y_coh)
plt.title('mean coherence')
plt.xlabel('n_topics // (10.000 posts)')
plt.ylabel('score')
plt.grid()
plt.show()

plt.plot(x_per, y_per)
plt.title('mean perplexity')
plt.xlabel('n_topics // (10.000 posts)')
plt.ylabel('score')
plt.grid()
plt.show()

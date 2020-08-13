import re
import time
import datetime
import pandas as pd
from tqdm import tqdm
from pymystem3 import Mystem

mystem = Mystem()
tqdm.pandas()

STOP_WORDS = {'которой', 'свою', 'своя', 'чтобы', 'моему', 'моё', 'такое', 'а', 'наш', 'сами', 'всю', 'такой', 'одну',
              'нам', 'во', 'нельзя', 'кто', 'имъ', 'можете', 'все', 'нашего', 'моём', 'чем', 'которое', 'у', 'почти',
              'будете', 'которым', 'еще', 'нашему', 'мой', 'вдруг', 'всею', 'собою', 'от', 'моими', 'ее', 'будем',
              'мочь', 'одних', 'тобой', 'ними', 'одного', 'тому', 'не', 'нашу', 'емъ', 'под', 'всем', 'тою', 'свои',
              'тех', 'я', 'между', 'наше', 'чём', 'ж', 'ни', 'таких', 'есть', 'и', 'ты', 'больше', 'это', 'опять',
              'более', 'такие', 'тут', 'самому', 'вот', 'вы', 'она', 'или', 'нашим', 'самого', 'тоже', 'вами', 'само',
              'иногда', 'кем', 'одною', 'будет', 'как', 'сама', 'оне', 'всеми', 'какая', 'ком', 'до', 'в', 'совсем',
              'которую', 'эти', 'нами', 'одному', 'свое', 'всея', 'ему', 'здесь', 'на', 'моей', 'сам', 'наша', 'весь',
              'своё', 'что', 'будьте', 'эту', 'вся', 'самим', 'моги', 'вам', 'нашими', 'своего', 'чего', 'такою',
              'этою', 'за', 'своими', 'но', 'для', 'этого', 'моим', 'то', 'моих', 'можем', 'всей', 'может', 'одни',
              'могли', 'тебе', 'них', 'самими', 'тобою', 'самих', 'саму', 'меня', 'уж', 'всему', 'ней', 'нею', 'так',
              'тебя', 'одним', 'себе', 'ко', 'могла', 'нибудь', 'себя', 'ем', 'чему', 'ешь', 'такая', 'него', 'ничего',
              'своем', 'много', 'ел', 'такому', 'ведь', 'сейчас', 'разве', 'мои', 'таком', 'могу', 'могите', 'котором',
              'своей', 'всегда', 'те', 'которых', 'нас', 'наши', 'этим', 'ест', 'хоть', 'да', 'которыми', 'там', 'ела',
              'моего', 'эта', 'томах', 'чуть', 'кому', 'этот', 'нем', 'были', 'своём', 'о', 'которая', 'они', 'будешь',
              'уже', 'мне', 'чтоб', 'моею', 'же', 'быть', 'из', 'нашей', 'одно', 'если', 'такими', 'всё', 'неё',
              'потом', 'нее', 'этих', 'одном', 'всех', 'три', 'впрочем', 'которому', 'наса', 'нашем', 'этом', 'где',
              'тогда', 'тот', 'был', 'который', 'никогда', 'бы', 'при', 'к', 'этому', 'этой', 'мною', 'нашею', 'хорошо',
              'мог', 'которою', 'зачем', 'теперь', 'была', 'моем', 'такую', 'лучше', 'над', 'об', 'будто', 'через',
              'того', 'своим', 'мной', 'наконец', 'конечно', 'мы', 'том', 'таким', 'со', 'с', 'буду', 'нему', 'без',
              'свой', 'ими', 'этими', 'та', 'ним', 'могут', 'когда', 'ту', 'которые', 'даже', 'про', 'ли', 'едим',
              'ещё', 'нём', 'мою', 'будучи', 'кого', 'одна', 'могло', 'будь', 'вас', 'её', 'одной', 'раз', 'одними',
              'едят', 'оно', 'тем', 'той', 'ею', 'после', 'только', 'моя', 'он', 'будут', 'такого', 'всего', 'по',
              'собой', 'потому', 'куда', 'какой', 'своему', 'самом', 'ну', 'надо', 'нет', 'своих', 'мое', 'можно',
              'наших', 'всём', 'своею', 'которого', 'им', 'один', 'два', 'их', 'комья', 'теми', 'было', 'его', 'перед',
              'ей', 'можешь', 'другой'}

BAN_CHARS = set('qwertyuiopasdfghjklzxcvbnm1234567890_!@$%^&*()+=')
GOOD_CHARS = set('йцукенгшщзхъёдлорпавыфячсмитьбю')


def preprocess_text(raw_text, min_word_len, exclude_hashtags, lemmatize):
    ban_chars = BAN_CHARS.copy()
    raw_text = raw_text.replace('#', ' #')
    if exclude_hashtags:
        ban_chars.add('#')
    words = re.findall('[\w#]+', raw_text.lower())
    words = [word for word in words
             if set(word) & ban_chars == set()
             and set(word) & GOOD_CHARS != set()
             and len(word) >= min_word_len
             and word not in STOP_WORDS]
    if lemmatize:
        return [i for i in mystem.lemmatize(' '.join(words)) if i not in {'\n', ' ', '#'}]
    return words


def timing(func):

    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print("Function {1} ran for {0} seconds".format(round(time.time() - t, 2), func.__name__))
        return res

    return wrapper


def load_data(file_path, text_len_threshold=100, start_date=None, end_date=None):
    data = pd.read_csv(file_path)[['description', 'published']]
    data['published'] = pd.to_datetime(data.published, format="%Y-%m-%d %H:%M:%S")
    if start_date:
        data = data[(data['published'] >= start_date) & (data['published'] < end_date)]
    data['description'] = data['description'].fillna('')
    data = data[data['description'].apply(lambda i: len(i)) >= text_len_threshold]
    data['lemmas'] = data['description'].progress_apply(lambda i: preprocess_text(raw_text=i,
                                                                                  min_word_len=4,
                                                                                  exclude_hashtags=False,
                                                                                  lemmatize=True))
    return data


def split_weeks(df: pd.DataFrame, day_shift: float):
    weeks = list()
    time_delta = df.published.max() - df.published.min()
    for i in range(int(time_delta.days // day_shift + 1)):
        start = df.published.min() + datetime.timedelta(days=i * day_shift)
        end = start + datetime.timedelta(days=7)
        weeks.append(df[(df['published'] >= start) & (df['published'] < end)])
    return weeks

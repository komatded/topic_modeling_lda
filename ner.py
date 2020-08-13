from deeppavlov import configs, build_model
import pandas as pd
import tqdm
import re

ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
data = pd.read_csv('../resources/post_children_aud.csv')

parsed, broken = list(), list()

for description in tqdm.tqdm(data.sample(50000).description.values):
    try:
        text = ' '.join(re.findall('\w+', description)[:100])
        parsed.append(ner_model([text]))
    except KeyboardInterrupt:
        quit()
    except:
        broken.append(description)


def get_entities(tokens, tags):
    result = list()
    entity, entity_tag = list(), str()
    for token, tag in zip(tokens, tags):
        if tag.startswith('B'):
            if len(entity) == 0:
                entity.append(token)
                entity_tag = tag.split('-')[1]
            else:
                result.append((' '.join(entity), entity_tag))
                entity = [token]
                entity_tag = tag.split('-')[1]
        elif tag.startswith('I'):
            entity.append(token)
        else:
            if len(entity) != 0:
                result.append((' '.join(entity), entity_tag))
                entity, entity_tag = list(), str()
    return result


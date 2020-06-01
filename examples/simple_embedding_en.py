from typing import List

import swem
from gensim.models import KeyedVectors


def tokenize_en(text: str) -> List[str]:
    text_processed = text.replace('.', ' .').replace(',', ' ,')
    return text_processed.replace('?', ' ?').replace('!', ' !').split()


if __name__ == '__main__':
    kv = KeyedVectors.load('wiki_mecab-ipadic-neologd.kv')
    swem_embed = swem.SWEM(kv, tokenizer=tokenize_en)

    doc: str = 'This is an implementation of SWEM.'
    embed = swem_embed.infer_vector(doc, method='max')
    print(embed)
    print(embed.shape)

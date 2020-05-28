from typing import List

import MeCab
import swem
from gensim.models import KeyedVectors


def tokenize_ja(text: str, args: str = '-O wakati') -> List[str]:
    tagger = MeCab.Tagger(args)
    return tagger.parse(text).strip().split(' ')


if __name__ == '__main__':
    model = KeyedVectors.load('wiki_mecab-ipadic-neologd.kv')
    swem_embed = swem.SWEM(model, tokenize_ja)

    doc = 'すもももももももものうち'
    embed = swem_embed.infer_vector(doc, method='max')
    print(embed)
    print(embed.shape)

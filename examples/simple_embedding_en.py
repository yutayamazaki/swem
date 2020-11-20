from typing import List

import swem
from gensim.models import KeyedVectors


def tokenize_en(text: str) -> List[str]:
    text_processed = text.replace('.', ' .').replace(',', ' ,')
    return text_processed.replace('?', ' ?').replace('!', ' !').split()


if __name__ == '__main__':
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    swem_embed = swem.SWEM(kv)

    tokens: List[str] = ['This', 'is', 'an', 'implementation', 'of', 'SWEM']
    embed = swem_embed.infer_vector(tokens, method='max')
    print(embed)
    print(embed.shape)

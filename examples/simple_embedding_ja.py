from typing import List

import swem
from gensim.models import KeyedVectors


if __name__ == '__main__':
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    swem_embed = swem.SWEM(kv)

    tokens: List[str] = ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
    embed = swem_embed.infer_vector(tokens, method='max')
    print(embed)
    print(embed.shape)

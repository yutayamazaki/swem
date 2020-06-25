from typing import List

import numpy as np
import swem
from gensim.models import KeyedVectors

if __name__ == '__main__':
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    tokens: List[str] = ['I', 'have', 'a', 'pen']

    embed: np.ndarray = swem.infer_vector(
        tokens=tokens, kv=kv, method='concat'
    )
    print(embed.shape)

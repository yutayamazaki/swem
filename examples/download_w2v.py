from typing import List

import numpy as np
import swem

if __name__ == '__main__':
    lang: str = 'ja'
    swem.download_w2v(lang=lang)
    kv = swem.load_w2v(lang=lang)

    tokens: List[str] = ['I', 'have', 'a', 'pen']

    embed: np.ndarray = swem.infer_vector(
        tokens=tokens, kv=kv, method='max'
    )
    print(embed.shape)

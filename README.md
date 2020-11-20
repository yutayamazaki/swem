# SWEM
![GitHub Actions](https://github.com/yutayamazaki/swem/workflows/build/badge.svg)
[![PyPI Version](https://img.shields.io/pypi/v/swem.svg)](https://pypi.org/project/swem/)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
![GitHub Starts](https://img.shields.io/github/stars/yutayamazaki/swem.svg?style=social)
![GitHub Forks](https://img.shields.io/github/forks/yutayamazaki/swem.svg?style=social)

Implementation of SWEM(Simple Word-Embedding-based Models)  
[Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms (ACL 2018)](https://arxiv.org/abs/1805.09843)

## Installation

```shell
pip install swem
```

## Example

Examples are available in [examples](https://github.com/yutayamazaki/swem/tree/main/examples) directory.


- [functional_api.py](https://github.com/yutayamazaki/swem/blob/main/examples/functional_api.py)
- [simple_embedding_en.py](https://github.com/yutayamazaki/swem/blob/main/examples/simple_embedding_en.py)
- [simple_embedding_ja.py](https://github.com/yutayamazaki/swem/blob/main/examples/simple_embedding_ja.py)


### Functional API

```python example.py
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

```


### Japanese

```python example.py
from typing import List

import swem
from gensim.models import KeyedVectors


if __name__ == '__main__':
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    swem_embed = swem.SWEM(kv)

    tokens: List[str] = ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
    embed = swem_embed.infer_vector(tokens, method='max')
    print(embed.shape)
```

Results
```shell
(200,)
```

### English

```python example.py
from typing import List

import swem
from gensim.models import KeyedVectors


if __name__ == '__main__':
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    swem_embed = swem.SWEM(kv)

    tokens: List[str] = ['This', 'is', 'an', 'implementation', 'of', 'SWEM']
    embed = swem_embed.infer_vector(tokens, method='max')
    print(embed.shape)
```

Results
```shell
(200,)
```


## Set random seed

SWEM generates random vector when given token is out of vocaburary. To reproduce token's embeddings, you need to set seed of NumPy.

```python
from typing import List

import numpy as np
import swem
from gensim.models import KeyedVectors

if __name__ == '__main__':
    np.random.seed(0)
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    tokens: List[str] = ['I', 'have', 'a', 'pen']

    embed: np.ndarray = swem.infer_vector(
        tokens=tokens, kv=kv, method='concat'
    )
    print(embed.shape)

```


## Download pretained w2v and use it.

```python
import swem
swem.download_w2v(lang='ja')
kv = swem.load_w2v(lang='ja')
```

```shell
Downloading w2v file to /Users/<username>/.swem/ja.zip
Extract zipfile into /Users/<username>/.swem/ja
Success to extract files.
```

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

Examples are available in [examples](https://github.com/yutayamazaki/swem/tree/master/examples) directory.  

- [simple_embedding_en.py](https://github.com/yutayamazaki/swem/blob/master/examples/simple_embedding_en.py)
- [simple_embedding_ja.py](https://github.com/yutayamazaki/swem/blob/master/examples/simple_embedding_ja.py)
- [use_custom_tokenizer_ja.py](https://github.com/yutayamazaki/swem/blob/master/examples/use_custom_tokenizer_ja.py)


### Japanese

```python example.py
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


def tokenize_en(text: str) -> List[str]:
    text_processed = text.replace('.', ' .').replace(',', ' ,')
    return text_processed.replace('?', ' ?').replace('!', ' !').split()


if __name__ == '__main__':
    model = KeyedVectors.load('wiki_mecab-ipadic-neologd.kv')
    swem_embed = swem.SWEM(model, tokenizer=tokenize_en)

    doc = 'This is an implementation of SWEM.'
    embed = swem_embed.infer_vector(doc, method='max')
    print(embed.shape)
```

Results
```shell
(200,)
```

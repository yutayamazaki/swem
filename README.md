# SWEM
![GitHub Actions](https://github.com/yutayamazaki/SWEM-Python/workflows/build/badge.svg)
[![PyPI Version](https://img.shields.io/pypi/v/swem.svg)](https://pypi.org/project/swem/)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
![GitHub Starts](https://img.shields.io/github/stars/yutayamazaki/SWEM-Python.svg?style=social)
![GitHub Forks](https://img.shields.io/github/forks/yutayamazaki/SWEM-Python.svg?style=social)

Implementation of SWEM(Simple Word-Embedding-based Models)  
[Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms (ACL 2018)](https://arxiv.org/abs/1805.09843)

## Installation

```shell
pip install swem
```

## Example

```python example.py
import swem
from gensim.models import Word2Vec

if __name__ == '__main__':
    model = Word2Vec.load('wiki_mecab-ipadic-neologd.model')
    swem_embed = swem.SWEM(model)

    doc = 'すもももももももものうち'
    embed = swem_embed.infer_vector(doc, method='average')
    print(embed.shape)
```

Results  
```shell
(200,)
```

# SWEM
Implementation of SWEM(Simple Word-Embedding-based Models)  
[Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms (ACL 2018)](https://arxiv.org/abs/1805.09843)

Details are available [here(Japanese)](https://scrapbox.io/whey-memo/SWEM(Simple_Word-Embedding-Based_Models)).

## Example

```python example.py
from gensim.models.word2vec import Word2Vec

from swem import SWEM

if __name__ == '__main__':
    model = Word2Vec.load('wiki_mecab-ipadic-neologd.model')

    swem = SWEM(model)

    doc = '僕の名前はバナナです。'


    for method in ['max', 'average', 'concat']:
        print(swem.infer_vector(doc, method=method).shape)
```

Results  
```shell
(200,)
(200,)
(400,)
```
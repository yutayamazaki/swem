from typing import List

import MeCab
import numpy as np


def tokenize(text: str) -> List[str]:
    tagger = MeCab.Tagger('-O wakati')
    return tagger.parse(text).strip().split(' ')


class SWEM:
    """Implementation of SWEM.

    Parameters
    ----------
    model: gensim.models.word2vec.Word2Vec
        Word2Vec model

    tokenizer: callable
        Callable object to tokenize input text.

    uniform_range: tuple of float
    """

    def __init__(self, model, tokenizer=None, uniform_range=(-0.01, 0.01)):
        self.model = model
        if tokenizer is None:
            tokenizer = tokenize
        self.tokenizer = tokenizer
        self.uniform_range = uniform_range
        self.embed_dim = self.model.wv.vector_size

    def _doc_embed(self, tokens: List[str]) -> np.ndarray:
        doc_embed = []
        for token in tokens:
            try:
                doc_embed.append(self.model.wv[token])
            except Exception as e:
                print(e)
                doc_embed.append(np.random.uniform(self.uniform_range[0],
                                                   self.uniform_range[1],
                                                   self.embed_dim))
        return np.array(doc_embed)

    @staticmethod
    def _hierarchical_pool(doc_embed: np.ndarray, n: int) -> np.ndarray:
        text_len = doc_embed.shape[0]
        if n > text_len:
            raise ValueError(f'window size [{n}] must be less '
                             f'than text length{text_len}.')

        n_iters = text_len - n + 1
        pooled_doc_embed = [
            np.mean(doc_embed[i:i + n], axis=0) for i in range(n_iters)
        ]
        return np.max(pooled_doc_embed, axis=0)

    def infer_vector(self, doc: str, method='max', n=3) -> np.ndarray:
        tokens = self.tokenizer(doc)
        doc_embed = self._doc_embed(tokens)

        if method == 'max':
            return doc_embed.max(axis=0)

        elif method == 'average':
            return doc_embed.mean(axis=0)

        elif method == 'concat':
            return np.hstack([doc_embed.mean(axis=0), doc_embed.max(axis=0)])

        elif method == 'hierarchical':
            return self._hierarchical_pool(doc_embed, n)

        else:
            raise AttributeError(
                f'infer_vector has no attribute [{method}] method.'
            )

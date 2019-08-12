from typing import List

import MeCab
import numpy as np


def tokenize(text: str) -> list:
    tagger = MeCab.Tagger('-O wakati')
    return tagger.parse(text).strip().split(' ')


class SWEM:

    def __init__(self, model, tokenizer=None, uniform_range=(-0.01, 0.01)):
        self.model = model
        if tokenizer is None:
            tokenizer = tokenize
        self.tokenizer = tokenizer
        self.uniform_range = uniform_range
        self.embed_dim = self.model.wv.vector_size

    def _doc_embed(self, doc: str) -> np.ndarray:
        doc_embed = []
        for word in self.tokenizer(doc):
            try:
                doc_embed.append(self.model.wv[word])
            except:
                doc_embed.append(np.random.uniform(self.uniform_range[0],
                                                   self.uniform_range[1],
                                                   self.embed_dim))
        return np.array(doc_embed)

    def infer_vector(self, doc, method='max'):
        doc_embed = self._doc_embed(doc)

        if method == 'max':
            return doc_embed.max(axis=0)

        elif method == 'average':
            return doc_embed.mean(axis=0)

        elif method == 'concat':
            return np.hstack([doc_embed.mean(axis=0), doc_embed.max(axis=0)])

        else:
            raise AttributeError(f'infer_vector has no attribute method={method}.')
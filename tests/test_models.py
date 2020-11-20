import unittest
from typing import Dict, List, Tuple

import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors, Word2VecKeyedVectors

import swem
from swem import models


def test_load_w2v_success():
    kv = swem.load_w2v(lang='ja')
    assert isinstance(kv, Word2VecKeyedVectors)


def test_load_w2v_invalid_lang():
    with pytest.raises(ValueError):
        swem.load_w2v(lang='invalid-lang')


def test_word_embed():
    token: str = '私'
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    embed = models._word_embed(token, kv=kv)
    assert embed.shape == (200, )


def test_word_embeds():
    tokens: List[str] = ['私', 'は']
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    embed = models._word_embeds(tokens, kv=kv, uniform_range=(-0.01, 0.01))
    assert embed.shape == (2, 200)


def test_hierarchical_pool():
    tokens: List[str] = ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
    kv: KeyedVectors = KeyedVectors(vector_size=200)

    word_embeds: np.ndarray = models._word_embeds(tokens, kv, (-1, 1))
    ret: np.ndarray = models._hierarchical_pool(word_embeds, num_windows=3)
    assert ret.shape == (200, )


def test_hierarchical_pool_raise():
    """ ValueError: when invalid n_windows passed. """
    doc: str = '桃'
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    word_embeds = models._word_embeds(doc, kv, (-1, 1))
    with pytest.raises(ValueError):
        # text_length: 1, n_windows: 3
        models._hierarchical_pool(word_embeds, num_windows=3)


def test_infer_vector_functional():
    tokens: List[str] = ['私', 'は', '私', 'は']
    kv: KeyedVectors = KeyedVectors(vector_size=200)
    methods: Dict[str, Tuple[int]] = {
        'avg': (200, ),
        'max': (200, ),
        'concat': (400, ),
        'hierarchical': (200, )
    }
    for method, shape in methods.items():
        embed: np.ndarray = swem.infer_vector(tokens, kv=kv, method=method)
        assert embed.shape == shape


class SWEMTests(unittest.TestCase):

    def setUp(self):
        kv: KeyedVectors = KeyedVectors(vector_size=200)
        self.swem = models.SWEM(kv)

    def test_infer_vector(self):
        methods = {
            'avg': 200,
            'concat': 400,
            'hierarchical': 200,
            'max': 200,
        }
        tokens: List[str] = ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
        for method_name, embed_dim in methods.items():
            ret = self.swem.infer_vector(tokens, method=method_name)
            assert ret.shape == (embed_dim, )

    def test_infer_vector_raise(self):
        tokens: List[str] = ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
        method = 'invalid method'
        with pytest.raises(ValueError):
            self.swem.infer_vector(tokens, method=method)

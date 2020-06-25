import unittest
from typing import Dict, List, Tuple

import MeCab
import numpy as np
import pytest

import swem
from swem import models


def tokenize_ja(text: str, args: str = '-O wakati') -> List[str]:
    tagger = MeCab.Tagger(args)
    return tagger.parse(text).strip().split(' ')


def test_word_embed():
    token = '私'
    kv = MockKV()
    embed = models._word_embed(token, kv=kv)
    assert embed.shape == (200, )


def test_word_embeds():
    tokens = ['私', 'は']
    kv = MockKV()
    embed = models._word_embeds(tokens, kv=kv, uniform_range=(-0.01, 0.01))
    assert embed.shape == (2, 200)


def test_hierarchical_pool():
    text: str = 'すもももももももものうち'
    kv: MockKV = MockKV()
    word_embeds: np.ndarray = models._word_embeds(text, kv, (-1, 1))
    ret: np.ndarray = models._hierarchical_pool(word_embeds, num_windows=3)
    assert ret.shape == (200, )


def test_hierarchical_pool_raise():
    """ ValueError: when invalid n_windows passed. """
    doc: str = '桃'
    kv: MockKV = MockKV()
    word_embeds = models._word_embeds(doc, kv, (-1, 1))
    with pytest.raises(ValueError):
        # text_length: 1, n_windows: 3
        models._hierarchical_pool(word_embeds, num_windows=3)


def test_infer_vector_functional():
    tokens = ['私', 'は', '私', 'は']
    kv = MockKV()
    methods: Dict[str, Tuple[int]] = {
        'avg': (200, ),
        'max': (200, ),
        'concat': (400, ),
        'hierarchical': (200, )
    }
    for method, shape in methods.items():
        embed: np.ndarray = swem.infer_vector(tokens, kv=kv, method=method)
        assert embed.shape == shape


class MockKV:

    vector_size = 200

    def __getitem__(self, item):
        return np.zeros(self.vector_size)


class SWEMTests(unittest.TestCase):

    def setUp(self):
        self.swem = models.SWEM(MockKV(), tokenize_ja)

    def init_tokenizer_is_not_callable(self):
        with pytest.raises(ValueError):
            models.SWEM(MockKV(), 'aaaa')

    def test_infer_vector(self):
        methods = {
            'avg': 200,
            'concat': 400,
            'hierarchical': 200,
            'max': 200,
        }
        doc = 'すもももももももものうち'
        for method_name, embed_dim in methods.items():
            ret = self.swem.infer_vector(doc, method=method_name)
            assert ret.shape == (embed_dim, )

    def test_infer_vector_raise(self):
        doc = 'すもももももももものうち'
        method = 'invalid method'
        with pytest.raises(ValueError):
            self.swem.infer_vector(doc, method=method)

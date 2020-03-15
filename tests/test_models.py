import unittest

import numpy as np
import pytest

from swem import models, tokenizers


def test_word_embed():
    token = '私'
    w2v = MockW2V()
    embed = models._word_embed(token, wv=w2v.wv)
    assert embed.shape == (200, )


def test_doc_embed():
    tokens = ['私', 'は']
    w2v = MockW2V()
    embed = models._doc_embed(tokens, wv=w2v.wv, uniform_range=(-0.01, 0.01))
    assert embed.shape == (2, 200)


class MockW2V:

    class MockWV:

        vector_size = 200

        def __getitem__(self, item):
            return np.zeros(self.vector_size)

    wv = MockWV()


class SWEMTests(unittest.TestCase):

    def setUp(self):
        self.swem = models.SWEM(MockW2V())

    def test_init_raise_lang(self):
        """ Chech ValueError for invalid lang passed. """
        with pytest.raises(ValueError):
            models.SWEM(MockW2V(), lang='es')

    def test_init_tokenizer_en(self):
        """ Passed lang='en', tokenizer equals to tokenize_en """
        model = models.SWEM(MockW2V(), lang='en')
        assert model.tokenizer == tokenizers.tokenize_en

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
        with pytest.raises(AttributeError):
            self.swem.infer_vector(doc, method=method)

    def test_hierarchical_pool(self):
        doc = 'すもももももももものうち'
        doc_embed = models._doc_embed(doc, self.swem.model.wv, (-1, 1))
        ret = self.swem._hierarchical_pool(doc_embed, n_windows=3)
        assert ret.shape == (200, )

    def test_hierarchical_pool_raise(self):
        """ ValueError: when invalid n_windows passed. """
        doc = '桃'
        doc_embed = models._doc_embed(doc, self.swem.model.wv, (-1, 1))
        with pytest.raises(ValueError):
            # text_length: 1, n_windows: 3
            self.swem._hierarchical_pool(doc_embed, n_windows=3)

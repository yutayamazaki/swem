import unittest

import numpy as np
import pytest

from swem import models


def test_tokenize():
    """ A simple test for swem.tokenize """
    tokens = models.tokenize('私はバナナです。')
    assert tokens == ['私', 'は', 'バナナ', 'です', '。']


class MockW2V:

    class MockWV:

        vector_size = 200

        def __getitem__(self, item):
            return np.zeros(self.vector_size)

    wv = MockWV()


class SWEMTests(unittest.TestCase):

    def setUp(self):
        self.swem = models.SWEM(MockW2V())

    def test_doc_embed(self):
        tokens = ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
        ret = self.swem._doc_embed(tokens)
        assert ret.shape == (7, 200)

    def test_infer_vector(self):
        methods = {
            'average': 200,
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
        doc_embed = self.swem._doc_embed(doc)
        ret = self.swem._hierarchical_pool(doc_embed, n=3)
        assert ret.shape == (200, )

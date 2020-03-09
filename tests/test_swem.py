import unittest

from swem import models


def test_tokenize():
    """ A simple test for swem.tokenize """
    tokens = models.tokenize('私はバナナです。')
    assert tokens == ['私', 'は', 'バナナ', 'です', '。']

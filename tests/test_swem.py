import unittest

from src import swem


def test_tokenize():
    """ A simple test for swem.tokenize """
    tokens = swem.tokenize('私はバナナです。')
    assert tokens == ['私', 'は', 'バナナ', 'です', '。']

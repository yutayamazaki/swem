from swem import tokenizers


def test_tokenize_ja():
    """ A simple test for swem.tokenize """
    tokens = tokenizers.tokenize_ja('私はバナナです。')
    assert tokens == ['私', 'は', 'バナナ', 'です', '。']


def test_tokenize_en():
    """ test for swem.models.tokenize_en """
    text = 'This, is an implementation of SWEM.'
    tokens = tokenizers.tokenize_en(text)
    expected = ['This', ',', 'is', 'an', 'implementation', 'of', 'SWEM', '.']
    assert tokens == expected

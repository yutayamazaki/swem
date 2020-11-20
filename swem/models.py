import os
from typing import Dict, List, Tuple

from gensim.models import KeyedVectors
import numpy as np

import swem


def load_w2v(lang: str = 'ja') -> KeyedVectors:
    """Load local KeyedVectors. If specified model does not exists,
       download pretrained KeyedVectors.
    Args:
        lang (str): Specify the language.
    Returns:
        (KeyedVectors): Loaded KeyedVectors.
    """
    if lang not in ('ja'):
        raise ValueError(f'swem.load_kv has no support lang={lang}.')
    file_mapping: Dict[str, str] = {
        'ja': 'wiki_mecab-ipadic-neologd.kv'
    }
    home_dir: str = os.path.expanduser('~')
    lang_dir: str = os.path.join(home_dir, '.swem', lang)
    path: str = os.path.join(lang_dir, file_mapping[lang])
    if not os.path.exists(path):
        swem.download_w2v(lang=lang)
    return KeyedVectors.load(path)


def _word_embed(
    token: str,
    kv: KeyedVectors,
    uniform_range: Tuple[float, float] = (-0.01, 0.01)
) -> np.ndarray:
    """ Get word embedding of given token.

    Args:
        token (str): A word token to get embed.
        kv (KeyedVectors): Vocabularies dictionary.
        uniform_range (Tuple[float, float]): A range of uniform distribution to
                                             generate random vector.

    Returns:
        numpy.ndarray: An array with shape (self.embed_dim, )
    """
    try:
        return kv[token]
    except Exception:
        embed_dim: int = kv.vector_size
        return np.random.uniform(
            uniform_range[0],
            uniform_range[1],
            embed_dim
        )


def _word_embeds(tokens: List[str], kv: KeyedVectors,
                 uniform_range: Tuple[float, float]) -> np.ndarray:
    """ Get word embeddings of given tokens.

    Args:
        tokens (List[str]): A word tokens to calculate embeddding.
        kv (KeyedVectors): A dictionary of vocabularies.
        uniform_range (Tuple[float, float]):
            A range of uniform distributioin to
            generate random vector.

    Returns:
        numpy.ndarray: An embedding array with shape
                       (token_size, self.embed_dim, ).
    """
    doc_embed: List[np.ndarray] = []
    for token in tokens:
        word_embed: np.ndarray = _word_embed(
            token=token, kv=kv, uniform_range=uniform_range
        )
        doc_embed.append(word_embed)
    return np.array(doc_embed)


def _hierarchical_pool(
    tokens_embed: np.ndarray,
    num_windows: int = 3
) -> np.ndarray:
    """ Hierarchical Pooling: It takes word-order or spatial information
        into consideration when calculate document embeddings.

    Args:
        tokens_embed (np.ndarray): An embeded document vector.
        num_windows (int): A sizw of window to consider sequence.

    Returns:
        numpy.ndarray: An embedding array with shape (self.embed_dim, ).
    """
    text_len: int = tokens_embed.shape[0]
    if num_windows > text_len:
        raise ValueError(f'window size [{num_windows}] must be less '
                         f'than text length{text_len}.')

    num_iters: int = text_len - num_windows + 1
    pooled_doc_embed: List[np.ndarray] = [
        np.mean(tokens_embed[i:i + num_windows],
                axis=0) for i in range(num_iters)
    ]
    return np.max(pooled_doc_embed, axis=0)


def infer_vector(
    tokens: List[str],
    kv: KeyedVectors,
    method: str = 'avg',
    uniform_range: Tuple[float, float] = (-0.01, 0.01),
    num_windows: int = 3
) -> np.ndarray:
    """Infer vector by swem with specified method.
    Args:
        tokens (List[str]): A list of tokens like ['I', 'am', 'a', 'pen'].
        kv (KeyedVectors): Vocabularies.
        method (str): One of them ('avg', 'max', 'concat', 'hierarchical').
        uniform_range (Tuple[float, float]): Value range of random vector.
        num_windows (int): A window size used in hierarchical pooling.
    Returns:
        np.ndarray: With shape (N,).
    """
    tokens_embed: np.ndarray = _word_embeds(
        tokens=tokens,
        kv=kv,
        uniform_range=uniform_range
    )

    if method == 'max':
        return tokens_embed.max(axis=0)

    elif method == 'avg':
        return tokens_embed.mean(axis=0)

    elif method == 'concat':
        return np.hstack([tokens_embed.mean(axis=0), tokens_embed.max(axis=0)])

    elif method == 'hierarchical':
        return _hierarchical_pool(tokens_embed, num_windows)

    else:
        raise ValueError(
            f'infer_vector has no attribute [{method}] method.'
        )


class SWEM:
    """Implementation of SWEM.

    Args:
        kv: gensim.models.keyedvectors.Word2VecKeyedVectors
        uniform_range: Tuple[float, ...]
            A range of uniform distribution to create random embedding.
    """
    def __init__(
        self, kv: KeyedVectors,
        uniform_range: Tuple[float, float] = (-0.01, 0.01)
    ):
        self.kv: KeyedVectors = kv
        self.uniform_range: Tuple[float, float] = uniform_range

    def infer_vector(
        self,
        tokens: List[str],
        method: str = 'max',
        num_windows: int = 3
    ) -> np.ndarray:
        """ A main method to get document vector.

        Args:
            tokens (List[str]): A list of tokens.
            method (str): Designate method to pool.
                         ('max', 'avg', 'concat', 'hierarchical')

        Returns:
            numpy.ndarray: An embedding array.
        """
        doc_embed: np.ndarray = _word_embeds(
            tokens=tokens,
            kv=self.kv,
            uniform_range=self.uniform_range
        )

        if method == 'max':
            return doc_embed.max(axis=0)

        elif method == 'avg':
            return doc_embed.mean(axis=0)

        elif method == 'concat':
            return np.hstack([doc_embed.mean(axis=0), doc_embed.max(axis=0)])

        elif method == 'hierarchical':
            return _hierarchical_pool(doc_embed, num_windows)

        else:
            raise ValueError(
                f'infer_vector has no attribute [{method}] method.'
            )

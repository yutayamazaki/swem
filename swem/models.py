from typing import List, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np

from swem import tokenizers


def _word_embed(
    token: str,
    kv: Word2VecKeyedVectors,
    uniform_range: Tuple[float, ...] = (-0.01, 0.01)
) -> np.ndarray:
    """ Get word embedding of given token.

    Args:
        token (str): A word token to embed.
        kv (Word2VecKeyedVectors): Vocabularies dictionary.
        uniform_range (Tuple[float, ...]): A range of uniform distribution to
                                           generate random vector.

    Returns:
        numpy.ndarray: An array with shape (self.embed_dim, )
    """
    try:
        return kv[token]
    except Exception as e:
        print(e)
        embed_dim = kv.vector_size
        return np.random.uniform(
            uniform_range[0],
            uniform_range[1],
            embed_dim
        )


def _word_embeds(tokens: List[str], kv: Word2VecKeyedVectors,
                 uniform_range: Tuple[float, ...]) -> np.ndarray:
    """ Get word embeddings of given tokens.

    Args:
        tokens (List[str]): A word tokens to calculate embeddding.

    Returns:
        numpy.ndarray: An embedding array with shape
                       (token_size, self.embed_dim, ).
    """
    doc_embed = []
    for token in tokens:
        word_embed: np.ndarray = _word_embed(
            token=token, kv=kv, uniform_range=uniform_range
        )
        doc_embed.append(word_embed)
    return np.array(doc_embed)


class SWEM:
    """Implementation of SWEM.

    Args:
        model: gensim.models.word2vec.Word2Vec
            Word2Vec model
        tokenizer: Callable
            Callable object to tokenize input text.
        uniform_range: Tuple[float, ...]
            A range of uniform distribution to create random embedding.
        lang (str): 'ja' or `en`. Default value is 'ja'.
    """

    def __init__(self, model, tokenizer=None, uniform_range=(-0.01, 0.01),
                 lang: str = 'ja'):
        self.model = model
        if tokenizer is None:
            if lang == 'ja':
                tokenizer = tokenizers.tokenize_ja
            elif lang == 'en':
                tokenizer = tokenizers.tokenize_en
            else:
                msg = f'Argument [lang] does not support: "{lang}".'
                raise ValueError(msg)
        self.tokenizer = tokenizer
        self.uniform_range: Tuple[float, ...] = uniform_range

    @staticmethod
    def _hierarchical_pool(
        doc_embed: np.ndarray,
        n_windows: int
    ) -> np.ndarray:
        """ Hierarchical Pooling: It takes word-order or spatial information
            into consideration when calculate document embeddings.

        Args:
            doc_embed (np.ndarray): An embeded document vector.
            n_windows (int): A sizw of window to consider sequence.

        Returns:
            numpy.ndarray: An embedding array with shape (self.embed_dim, ).
        """
        text_len: int = doc_embed.shape[0]
        if n_windows > text_len:
            raise ValueError(f'window size [{n_windows}] must be less '
                             f'than text length{text_len}.')

        n_iters: int = text_len - n_windows + 1
        pooled_doc_embed: List[np.ndarray] = [
            np.mean(doc_embed[i:i + n_windows],
                    axis=0) for i in range(n_iters)
        ]
        return np.max(pooled_doc_embed, axis=0)

    def infer_vector(
        self,
        doc: str,
        method: str = 'max',
        n_windows: int = 3
    ) -> np.ndarray:
        """ A main method to get document vector.

        Args:
            doc (str): A document str to get embeddings.
            method (str): Designate method to pool.
                         ('max', 'avg', 'concat', 'hierarchical')

        Returns:
            numpy.ndarray: An embedding array.
        """
        tokens: List[str] = self.tokenizer(doc)
        doc_embed: np.ndarray = _word_embeds(
            tokens=tokens,
            kv=self.model.wv,
            uniform_range=self.uniform_range
        )

        if method == 'max':
            return doc_embed.max(axis=0)

        elif method == 'avg':
            return doc_embed.mean(axis=0)

        elif method == 'concat':
            return np.hstack([doc_embed.mean(axis=0), doc_embed.max(axis=0)])

        elif method == 'hierarchical':
            return self._hierarchical_pool(doc_embed, n_windows)

        else:
            raise AttributeError(
                f'infer_vector has no attribute [{method}] method.'
            )

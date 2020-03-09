from gensim.models.word2vec import Word2Vec

from swem import SWEM

if __name__ == '__main__':
    model = Word2Vec.load('wiki_mecab-ipadic-neologd.model')
    swem = SWEM(model)

    doc = '私の名前はバナナだ。'
    for method in ['max', 'average', 'concat', 'hierarchical']:
        print(swem.infer_vector(doc, method=method).shape)

from gensim.models.word2vec import Word2Vec

from swem import SWEM

if __name__ == '__main__':
    model = Word2Vec.load('wiki_mecab-ipadic-neologd.model')

    swem = SWEM(model)

    doc = '吾輩は猫である。名前はまだ無い。'

    for method in ['max', 'average', 'concat']:
        print(swem.infer_vector(doc, method=method).shape)
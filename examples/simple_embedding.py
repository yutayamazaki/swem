import swem

from gensim.models import Word2Vec

if __name__ == '__main__':
    model = Word2Vec.load('wiki_mecab-ipadic-neologd.model')
    swem_embed = swem.SWEM(model)

    doc = 'すもももももももものうち'
    embed = swem_embed.infer_vector(doc, method='average')
    print(embed)
    print(embed.shape)

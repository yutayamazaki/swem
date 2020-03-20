import swem

from gensim.models import KeyedVectors

if __name__ == '__main__':
    model = KeyedVectors.load('wiki_mecab-ipadic-neologd.kv')
    swem_embed = swem.SWEM(model)

    doc = 'すもももももももものうち'
    embed = swem_embed.infer_vector(doc, method='max')
    print(embed)
    print(embed.shape)

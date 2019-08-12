import glob
import os


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_livedoor(root='./text'):
    dirs = {
        0: 'dokujo-tsushin',
        1: 'it-life-hack',
        2: 'kaden-channel',
        3: 'livedoor-homme',
        4: 'movie-enter',
        5: 'peachy',
        6: 'smax',
        7: 'sports-watch',
        8: 'topic-news'
    }
    texts = []
    labels = []
    for label, d in dirs.items():
        files = glob.glob(os.path.join(root, d, '*'))
        for file in files:
            if os.path.basename(file) == 'LICENSE.txt':
                continue
            texts.append(load_text(file))
            labels.append(label)
    return texts, labels
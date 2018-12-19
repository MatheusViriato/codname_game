import gensim
from codname_functions import expand_tuple, show_results


def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True, limit=500000
    )

    board = {
        'blue': ['spell', 'lock', 'charge', 'tail', 'link'],
        'red': [
            'cat', 'button', 'pipe', 'pants',
            'mount', 'sleep', 'stick', 'file', 'worm'
        ],
        'assassin': 'doctor'
    }

    similar_words = model.most_similar(
        positive=board['red'],
        negative=board['blue'].append(board['assassin']),
        restrict_vocab=50000
    )

    words, score = expand_tuple(similar_words)

    show_results(words, score)


if __name__ == "__main__":
    main()

import gensim
from codname_functions import expand_tuple, show_results


def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True, limit=500000
    )

    word = input('Enter a word: ')
    limit = input('Set a limit: ')

    similar_words = model.similar_by_word(word, topn=int(limit))
    words, score = expand_tuple(similar_words)

    show_results(words, score, word)


if __name__ == "__main__":
    main()

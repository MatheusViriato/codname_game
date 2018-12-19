# -*- coding: utf-8 -*-

import operator
import gensim
import numpy as np
import json
from sklearn import linear_model


def verifyTwoWords(candidate, word_list):
    if '_' in candidate:
        return False
    return [word in candidate or candidate in word for word in word_list]


def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True, limit=200000)
    my_words = ['ambulance', 'hospital', 'spell',
                'lock', 'charge', 'tail', 'link', 'cook', 'web']
    enemy_words = ['smuggler', 'crown', 'cotton',
                   'palm', 'pumpkin', 'giant', 'link', 'dog']
    assassin = 'tie'

    X = np.array([model.word_vec(word)
                  for word in my_words + enemy_words + [assassin]])
    Y = np.array([1 for i in range(len(my_words))] +
                 [-1 for i in range(len(enemy_words)+1)])

    clf = linear_model.SGDClassifier(
        max_iter=1000, fit_intercept=False, penalty='none')
    clf.fit(X, Y)

    vec_words = clf.coef_[0]
    clf_similar_words = model.similar_by_vector(vec_words, topn=100)

    clf_result = []

    for word_candidate, _ in clf_similar_words:
        if verifyTwoWords(word_candidate, my_words):
            scores = {}
            for word in my_words:
                scores[word] = model.similarity(word_candidate, word)
            clf_result.append({word_candidate: [w for w, s in sorted(
                scores.items(), key=operator.itemgetter(1), reverse=True)[:5]]})

    with open('result.json', 'w') as outfile:
        json.dump(clf_result, outfile)


if __name__ == "__main__":
    main()

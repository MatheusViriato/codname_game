import pandas as pd
import matplotlib.pyplot as plt


def expand_tuple(similar_words):
    return [x[0] for x in similar_words], [x[1] for x in similar_words]


def show_results(words, score, word=""):
    data = {}
    word_list = []

    [word_list.append({'word': f, 'score': b}) for f, b in zip(words, score)]

    data['result'] = word_list
    df = pd.DataFrame(data['result'])

    axs = df.sort_values(by=['score']).plot(
        x='word', kind='barh', y='score', colormap='Paired')
    if word != "":
        axs.set_title('Similar words for "' + word + '"')
    else:
        axs.set_title('Similar words for your game')
    axs.set_ylabel('Score for words')

    plt.show()

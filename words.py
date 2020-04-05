import csv

STOP_WORD = "</S>"
START_WORD = "<S>"


def _load_words(path="/data/word_counts.txt", delimiter=' '):
    with open(path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        frequency_mapping = {row[0]: int(row[1]) for row in csv_reader}

    sorted_words = sorted(frequency_mapping.items(),
                          key=lambda kv: kv[1],
                          reverse=True)
    index2word = {k: v[0] for k, v in enumerate(sorted_words)}
    # index2word[11519] = '<pad>'

    return index2word


def _reverse_dictionary(dict):
    return {v: k for k, v in dict.items()}


_index_to_word = _load_words()
_word_to_index = _reverse_dictionary(_load_words())


def get_index(word):
    return _word_to_index[word]


def get_word(index):
    return _index_to_word[index]


def idx2sentence(indices):
    sentence = []
    for idx in indices:
        if idx != 11519:
            sentence.append(get_word(idx))
    return sentence


def format_mscoco(indices):
    sentence = idx2sentence(indices)
    sentence = sentence[1:-1]  # get rid of start and stop words
    sentence = ' '.join(sentence)
    sentence = sentence.replace('.', '')  # vielleicht auch Komma
    return sentence

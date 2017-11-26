import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def labels2seq(word2type, all_words, word_list, is_train):
    ls = [-1, 0, 1, 19, 20, 21, 22, 41, 42, 43, 44, 45, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
          134, 135, 136, 137,
          138, 139, 140, 141, 142, 143, 146, 147, 148, 149, 150, 250, 251, 252, 253, 350, 354, 355, 356, 357, 358, 359,
          360, 462, 1000]
    seqs = [[1, 41, 42], [1, 41, 43], [1, 41, 44], [1, 41, 45], [1, 19], [1, 20], [1, 21],
            [146, 147], [146, 148], [146, 149], [146, 150], [131, 132], [131, 133], [131, 134], [131, 135],
            [131, 136], [131, 137], [131, 138], [131, 139], [462], [22], [121, 122],
            [121, 123], [121, 124], [250, 251], [250, 252], [250, 253], [140, 141], [140, 142], [140, 143],
            [125, 126], [125, 127, 128], [125, 127, 129], [125, 127, 130], [350, 354], [350, 355],
            [350, 356], [350, 357], [350, 358], [350, 359], [350, 360]]
    targets = [[2, 7, 8, 53], [2, 7, 9, 53], [2, 7, 10, 53], [2, 7, 11, 53], [2, 3, 53, 0], [2, 4, 53, 0],
               [2, 5, 53, 0], [35, 36, 53, 0],
               [35, 37, 53, 0], [35, 38, 53, 0], [35, 39, 53, 0], [22, 23, 53, 0], [22, 24, 53, 0], [22, 25, 53, 0],
               [22, 26, 53, 0],
               [22, 27, 53, 0], [22, 28, 53, 0], [22, 29, 53, 0], [22, 30, 53, 0], [52, 53, 0, 0], [6, 53, 0, 0],
               [12, 13, 53, 0],
               [12, 14, 53, 0], [12, 15, 53, 0], [40, 41, 53, 0], [40, 42, 53, 0], [40, 43, 53, 0], [31, 32, 53, 0],
               [31, 33, 53, 0],
               [31, 34, 53, 0], [16, 17, 53, 0], [16, 18, 19, 53], [16, 18, 20, 53], [16, 18, 21, 53], [44, 45, 53, 0],
               [44, 46, 53, 0],
               [44, 47, 53, 0], [44, 48, 53, 0], [44, 49, 53, 0], [44, 50, 53, 0], [44, 51, 53, 0]]
    d_inputs = [[2, 7, 8], [2, 7, 9], [2, 7, 10], [2, 7, 11], [2, 3, 0], [2, 4, 0], [2, 5, 0], [35, 36, 0], [35, 37, 0],
                [35, 38, 0],
                [35, 39, 0], [22, 23, 0], [22, 24, 0], [22, 25, 0], [22, 26, 0], [22, 27, 0], [22, 28, 0], [22, 29, 0],
                [22, 30, 0],
                [52, 0, 0], [6, 0, 0], [12, 13, 0], [12, 14, 0], [12, 15, 0], [40, 41, 0], [40, 42, 0], [40, 43, 0],
                [31, 32, 0],
                [31, 33, 0], [31, 34, 0], [16, 17, 0], [16, 18, 19], [16, 18, 20], [16, 18, 21], [44, 45, 0],
                [44, 46, 0], [44, 47, 0],
                [44, 48, 0], [44, 49, 0], [44, 50, 0], [44, 51, 0]]

    words = []
    label_seq = []
    vectors = []
    seq_len = []
    decoder_inputs = []
    cog = {132, 133, 134, 135, 136, 138, 139, 137}
    mlb = MultiLabelBinarizer()
    mlb.fit([ls])

    def _process(word, seq, target, decoder_input):
        words.append(all_words.index(word))
        vectors.append(all_words.index(word))
        label_seq.append(target)
        decoder_inputs.append([1] + decoder_input)  # append start sign
        seq_len.append(len(seq) + 1)

    if is_train:
        for word in word_list:
            labels = word2type[word]
            labels = set(labels)
            for seq, target, decoder_input in list(zip(seqs, targets, d_inputs)):
                if labels.issuperset(seq):
                    _process(word, seq, target, decoder_input)
        for word in word_list:
            if 121 in word2type[word] and 122 not in word2type[word] and 123 not in word2type[word] and 124 not in \
                    word2type[word]:
                seq = [121]
                decoder_input = [12, 0, 0]
                target = [12, 53, 0, 0]
                _process(word, seq, target, decoder_input)
            if 125 in word2type[word] and 126 not in word2type[word] and 127 not in word2type[word]:
                seq = [125]
                decoder_input = [16, 0, 0]
                target = [16, 53, 0, 0]
                _process(word, seq, target, decoder_input)
            if 127 in word2type[word] and 128 not in word2type[word] and 129 not in word2type[word] and 130 not in \
                    word2type[word]:
                seq = [125, 127]
                decoder_input = [16, 18, 0]
                target = [16, 18, 53, 0]
                _process(word, seq, target, decoder_input)
            if 131 in word2type[word] and len(cog.intersection(word2type[word])) == 0:
                seq = [131]
                decoder_input = [22, 0, 0]
                target = [22, 53, 0, 0]
                _process(word, seq, target, decoder_input)
            if 140 in word2type[word] and 141 not in word2type[word] and 142 not in word2type[word] and 143 not in \
                    word2type[word]:
                seq = [140]
                decoder_input = [31, 0, 0]
                target = [31, 53, 0, 0]
                _process(word, seq, target, decoder_input)
            if 146 in word2type[word] and 147 not in word2type[word] and 148 not in word2type[word] and 149 not in \
                    word2type[
                        word] and 150 not in word2type[word]:
                seq = [146]
                decoder_input = [35, 0, 0]
                target = [35, 53, 0, 0]
                _process(word, seq, target, decoder_input)
            if 250 in word2type[word] and 251 not in word2type[word] and 252 not in word2type[word] and 253 not in \
                    word2type[word]:
                seq = [250]
                decoder_input = [40, 0, 0]
                target = [40, 53, 0, 0]
                _process(word, seq, target, decoder_input)
    else:
        for word in word_list:
            words.append(all_words.index(word))
            vectors.append(all_words.index(word))
            label_seq.append(word2type[word])
            decoder_inputs.append(np.zeros(4, dtype=np.int32))  # append start sign
            seq_len.append(0)
        label_seq = mlb.fit_transform(label_seq)
    return np.array(words), np.array(vectors), np.array(label_seq), np.array(decoder_inputs), np.array(seq_len)


def prediction_with_threshold(t_preds, t_scores, threshold):
    t_preds[t_preds == -1] = 0
    new_preds = []
    t_preds = t_preds.transpose((0, 2, 1))
    for i in range(t_preds.shape[0]):
        single = []
        for j in range(t_preds.shape[1]):
            if t_scores[i, j] > threshold:
                single += t_preds[i, j].tolist()
        new_preds.append(single)
    return new_preds

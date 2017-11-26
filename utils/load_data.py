import io
import numpy as np
import re
from collections import defaultdict


def load_liwc(filename, word2vec, encoding='utf-8'):
    liwc_file = io.open(filename, 'r', encoding=encoding)
    lines = liwc_file.readlines()
    type2name = dict()
    word2type = dict()
    type2word = dict()
    lc = 0
    for i, line in enumerate(lines):  # read type
        if '%' in line:
            lc = i
            break
        tmp = line.strip().split()
        type2name[int(tmp[0])] = tmp[1]
    for line in lines[lc + 1:]:
        tmp = line.strip().split()
        if tmp[0] not in word2vec:
            continue
        word2type[tmp[0]] = list(map(int, tmp[1:]))
        for t in word2type[tmp[0]]:
            type2word[t] = type2word.get(t, [])
            type2word[t].append(tmp[0])
    return type2name, word2type, type2word


def load_vectors(filename, encoding='utf-8'):
    vector_file = io.open(filename, 'r', encoding=encoding)
    lines = vector_file.readlines()  # [:n]
    word2vector = dict()
    vector_size = int(lines[0].split()[1])
    for line in lines[1:]:
        line = line.strip()
        tmp = line.split()
        word2vector[tmp[0]] = list(map(float, tmp[1:]))
    vector_file.close()
    return vector_size, word2vector


def train_test(words, type2word, word2type, percent=0.2):
    np.random.seed(0)
    train = set()
    test = set()
    type_size = dict()
    for k, v in type2word.items():
        type_size[k] = len(v)
    np.random.shuffle(words)
    for word in words:
        if word in train or word in test:
            continue
        else:
            flag = 0
            for typei in word2type[word]:
                if type_size[typei] <= percent * len(type2word[typei]):
                    flag = 1
                    break
            if flag == 1:
                test.add(word)
            else:
                train.add(word)
                for typei in word2type[word]:
                    type_size[typei] -= 1
    return list(train), list(test)


def load_hownet(filename, word2type, word2vectors, encoding='utf-8'):
    hownet = open(filename, encoding=encoding)
    lines = hownet.readlines()
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    word2sememes = defaultdict(set)
    word2word_semevec = defaultdict(list)
    for line in lines:
        if 'W_C=' in line and line[4:].strip() in word2type:
            word = line[4:].strip()
        elif 'W_E=' in line and line[4:].strip() in word2type:
            word = line[4:].strip()
        elif 'DEF=' in line:
            m = re_words.findall(line, 0)
            word2sememes[word].update(m)
    word2sememes = dict(word2sememes)
    word2average_sememes = dict()
    biggest = 40
    word2sememe_length = dict()
    for word in word2type:
        word2word_semevec[word].append(word2vectors[word])
        word2sememe_length[word] = 1
        if word in word2sememes:
            count = 0
            ave = np.zeros(len(word2vectors[word]), np.float32)
            for sememe in word2sememes[word]:
                if sememe in word2vectors:
                    word2word_semevec[word].append(word2vectors[sememe])
                    count += 1
                    ave += np.array(word2vectors[sememe])
            word2word_semevec[word] += [[0] * len(word2vectors[word])] * (biggest - count)  # pad zero
            word2sememe_length[word] += count
            if count == 0:
                word2average_sememes[word] = ave
            else:
                word2average_sememes[word] = ave / count
        else:
            word2word_semevec[word] += [[0] * len(word2vectors[word])] * biggest  # pad 0
            word2average_sememes[word] = np.zeros(len(word2vectors[word]), np.float32)
    return dict(word2word_semevec), word2sememe_length, word2average_sememes
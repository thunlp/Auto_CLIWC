from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq import BasicDecoder, sequence_loss, GreedyEmbeddingHelper, dynamic_decode, TrainingHelper, \
    ScheduledEmbeddingTrainingHelper, tile_batch, BeamSearchDecoder, BahdanauAttention, AttentionWrapper
from utils import data_transform, load_data
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
import pickle

# Parameters
learning_rate = 1e-3
training_iters = 2000
display_step = 50
embedding_size = 300
beam_width = 5
EOS = 53
PAD = 0
GO = 1
max_seq_length = 4
sememe_size = 1617
# Network Parameters
n_hidden = 300  # hidden layer num of features
n_classes = 54  # 51+3

# files
vector_file = 'datasets/300Tvectors.txt'
liwc_file = 'datasets/sc_liwc.dic'
hownet_file = 'datasets/smallerHowNet.txt'

# data process
vector_size, word2vectors = load_data.load_vectors(vector_file)
type2name, word2type, type2word = load_data.load_liwc(liwc_file, word2vectors)
word2sememe_vecs, word2sememe_length, word2average_sememes = load_data.load_hownet(hownet_file, word2type, word2vectors)
percent = 0.2
np.random.seed(0)
train_list, test_list = load_data.train_test(list(word2type.keys()), type2word, word2type, percent)
pickle.dump(train_list, open('train.bin', 'wb'))
pickle.dump(test_list, open('test.bin', 'wb'))
test_len = len(test_list)
all_words = train_list + test_list
vector_matrix = list(map(lambda word: word2vectors[word], all_words))
memory_matrix = list(map(lambda word: word2sememe_vecs[word], all_words))
memory_lengths = list(map(lambda word: word2sememe_length[word], all_words))
ave_sememes = list(map(lambda word: word2average_sememes[word].tolist(), all_words))
words_index, vectors_index, label_seq, decoder_inputs, seq_len = data_transform.labels2seq(word2type, all_words, train_list, True)
t_words_index, t_vectors_index, t_label_seq, t_decoder_inputs, t_seq_len = data_transform.labels2seq(word2type, all_words, test_list, False)
train_batch_size = len(words_index)
test_batch_size = test_len
mlb = MultiLabelBinarizer()
train_len = len(words_index)
mlb.fit([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
          34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
layer1 = np.array([2, 12, 16, 22, 31, 35, 40, 44, 52, 6])
layer2 = np.array(
    [3, 4, 5, 7, 13, 14, 15, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51])
layer3 = np.array([8, 9, 10, 11, 19, 20, 21])

# tf Graph input
index_holder = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None, max_seq_length])
seq_length = tf.placeholder(tf.int32, [None])
decoder_inputs_h = tf.placeholder(tf.int32, shape=[None, max_seq_length])
keep_prob = tf.placeholder(tf.float32)

word_embeddings = tf.constant(vector_matrix, dtype=tf.float32)
sememe_embeddings = tf.constant(memory_matrix, dtype=tf.float32)
sememe_lengths = tf.constant(memory_lengths, dtype=tf.int32)
x = tf.gather(word_embeddings, index_holder)
sememe_memory = tf.gather(sememe_embeddings, index_holder)
memory_length = tf.gather(sememe_lengths, index_holder)
ave_sememes_embeddings = tf.constant(ave_sememes, dtype=tf.float32)
first_attention = tf.gather(ave_sememes_embeddings, index_holder)


def decoder(x, decoder_inputs, keep_prob, sequence_length, memory, memory_length, first_attention):
    with tf.variable_scope("Decoder") as scope:
        label_embeddings = tf.get_variable(name="embeddings", shape=[n_classes, embedding_size], dtype=tf.float32)
        train_inputs_embedded = tf.nn.embedding_lookup(label_embeddings, decoder_inputs)
        lstm = rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=keep_prob)
        output_l = layers_core.Dense(n_classes, use_bias=True)
        encoder_state = rnn.LSTMStateTuple(x, x)
        attention_mechanism = BahdanauAttention(embedding_size, memory=memory, memory_sequence_length=memory_length)
        cell = AttentionWrapper(lstm, attention_mechanism, output_attention=False)
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=train_batch_size)
        cell_state = cell_state.clone(cell_state=encoder_state, attention=first_attention)
        train_helper = TrainingHelper(train_inputs_embedded, sequence_length)
        train_decoder = BasicDecoder(cell, train_helper, cell_state, output_layer=output_l)
        decoder_outputs_train, decoder_state_train, decoder_seq_train = dynamic_decode(train_decoder, impute_finished=True)
        tiled_inputs = tile_batch(memory, multiplier=beam_width)
        tiled_sequence_length = tile_batch(memory_length, multiplier=beam_width)
        tiled_first_attention = tile_batch(first_attention, multiplier=beam_width)
        attention_mechanism = BahdanauAttention(embedding_size, memory=tiled_inputs, memory_sequence_length=tiled_sequence_length)
        x2 = tile_batch(x, beam_width)
        encoder_state2 = rnn.LSTMStateTuple(x2, x2)
        cell = AttentionWrapper(lstm, attention_mechanism, output_attention=False)
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=test_batch_size * beam_width)
        cell_state = cell_state.clone(cell_state=encoder_state2, attention=tiled_first_attention)
        infer_decoder = BeamSearchDecoder(cell, embedding=label_embeddings, start_tokens=[GO] * test_len, end_token=EOS,
                                          initial_state=cell_state, beam_width=beam_width, output_layer=output_l)
        decoder_outputs_infer, decoder_state_infer, decoder_seq_infer = dynamic_decode(infer_decoder, maximum_iterations=4)
        return decoder_outputs_train, decoder_outputs_infer, decoder_state_infer


decoder_outputs_train, decoder_outputs_infer, decoder_state_infer = decoder(x, decoder_inputs_h, keep_prob, seq_length, sememe_memory,
                                                                            memory_length, first_attention)
weights = tf.sequence_mask(seq_length, max_seq_length, dtype=tf.float32)
cost = sequence_loss(logits=decoder_outputs_train.rnn_output, targets=y, weights=weights)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# predicted_ids: [batch_size, sequence_length, beam_width]
pred = decoder_outputs_infer.predicted_ids
scores = decoder_state_infer.log_probs
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_iters:
        sess.run(optimizer, feed_dict={index_holder: vectors_index, y: label_seq, decoder_inputs_h: decoder_inputs, keep_prob: 0.5,
                                       seq_length: seq_len})
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={index_holder: vectors_index, y: label_seq, decoder_inputs_h: decoder_inputs,
                                             keep_prob: 0.5, seq_length: seq_len})
            print("Iter " + str(step) + ", batch Loss= " + str(loss))
            print('test')
            t_preds, t_scores = sess.run([pred, scores], feed_dict={index_holder: t_vectors_index, keep_prob: 1.0})
            t_preds_trans = data_transform.prediction_with_threshold(t_preds, t_scores, threshold=-1.6)
            t_preds_b = mlb.transform(t_preds_trans)
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq, t_preds_b[:, 2:53], average='micro')
            print('micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq, t_preds_b[:, 2:53], average='weighted')
            print('macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer1 - 2], t_preds_b[:, layer1],
                                                                              average='micro')
            print('layer1 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer1 - 2], t_preds_b[:, layer1],
                                                                              average='weighted')
            print('layer1 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer2 - 2], t_preds_b[:, layer2],
                                                                              average='micro')
            print('layer2 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer2 - 2], t_preds_b[:, layer2],
                                                                              average='weighted')
            print('layer2 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer3 - 2], t_preds_b[:, layer3],
                                                                              average='micro')
            print('layer3 micro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
            precisions, recalls, fscores, _ = precision_recall_fscore_support(t_label_seq[:, layer3 - 2], t_preds_b[:, layer3],
                                                                              average='weighted')
            print('layer3 macro average precision recall f1-score: %f %f %f' % (precisions, recalls, fscores))
        step += 1
    print("Optimization Finished!")

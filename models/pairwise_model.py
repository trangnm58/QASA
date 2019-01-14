import numpy as np
from keras.layers import Input

from models.graphs import *
from models.deep_model import DeepModel
from dataset import pad_sequences
from constants import PARAMS


class PairwiseModel(DeepModel, CharLSTM, ContextualLSTM):
    def __init__(self, model_name, embeddings, num_chars, batch_size, early_stopping, k_neg, fast_sample):
        CharLSTM.__init__(self)
        ContextualLSTM.__init__(self)
        DeepModel.__init__(self, model_name, batch_size, early_stopping, fast_sample)

        self.embeddings = embeddings
        self.num_chars = num_chars
        self.k_neg = k_neg

        self.embed_size = PARAMS['embed_size']
        self.keep_prob = PARAMS['keep_prob']

    def _add_placeholders(self):
        self.cp_word_ids = Input(name="cp_word_ids", shape=[None], dtype='int32')
        self.cp_char_ids = Input(name="cp_char_ids", shape=[None, None], dtype='int32')
        self.cp_sequence_lengths = Input(name='cp_sequence_lengths', batch_shape=[None], dtype='int32')
        self.cp_word_lengths = Input(name="cp_word_lengths", shape=[None], dtype='int32')

        self.cn_word_ids = Input(name="cn_word_ids", shape=[None], dtype='int32')
        self.cn_char_ids = Input(name="cn_char_ids", shape=[None, None], dtype='int32')
        self.cn_sequence_lengths = Input(name='cn_sequence_lengths', batch_shape=[None], dtype='int32')
        self.cn_word_lengths = Input(name="cn_word_lengths", shape=[None], dtype='int32')

        self.q_word_ids = Input(name="q_word_ids", shape=[None], dtype='int32')
        self.q_char_ids = Input(name="q_char_ids", shape=[None, None], dtype='int32')
        self.q_sequence_lengths = Input(name='q_sequence_lengths', batch_shape=[None], dtype='int32')
        self.q_word_lengths = Input(name="q_word_lengths", shape=[None], dtype='int32')

        self.keep_op = tf.placeholder(dtype=tf.float32, shape=[], name="keep_op")
        self.is_training = tf.placeholder(tf.bool, name='phase')

    def _add_model_op(self):
        CharLSTM._add_model_op(self)
        ContextualLSTM._add_model_op(self)

        self.word_embeddings = tf.Variable(initial_value=self.embeddings, name="word_embeddings",
                                           dtype=tf.float32, trainable=False)
        self.q_word_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.q_word_ids,
                                                        name="q_word_embeddings")
        self.cn_word_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.cn_word_ids,
                                                         name="cn_word_embeddings")
        self.cp_word_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.cp_word_ids,
                                                         name="cp_word_embeddings")

        self.n_score = None
        self.p_score = None

    def _add_train_op(self):
        margin = 1.0

        self.loss_op = tf.reduce_mean(tf.maximum(0., margin - self.p_score + self.n_score))

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0

        while idx < num_batch:
            example_batch = data[start:start + self.batch_size]
            if not example_batch:
                break

            q_batch, cp_batch, cn_batch = zip(*example_batch)

            q_char_ids, q_word_ids = zip(*[zip(*zip(*x)) for x in q_batch])
            padded_q_word_ids, q_sequence_lengths = pad_sequences(q_word_ids, pad_tok=0)
            padded_q_char_ids, q_word_lengths = pad_sequences(q_char_ids, pad_tok=0, nlevels=2)

            cp_char_ids, cp_word_ids = zip(*[zip(*zip(*x)) for x in cp_batch])
            padded_cp_word_ids, cp_sequence_lengths = pad_sequences(cp_word_ids, pad_tok=0)
            padded_cp_char_ids, cp_word_lengths = pad_sequences(cp_char_ids, pad_tok=0, nlevels=2)

            cn_char_ids, cn_word_ids = zip(*[zip(*zip(*x)) for x in cn_batch])
            padded_cn_word_ids, cn_sequence_lengths = pad_sequences(cn_word_ids, pad_tok=0)
            padded_cn_char_ids, cn_word_lengths = pad_sequences(cn_char_ids, pad_tok=0, nlevels=2)

            start += self.batch_size
            idx += 1
            yield (padded_q_word_ids, padded_q_char_ids, q_sequence_lengths, q_word_lengths,
                   padded_cp_word_ids, padded_cp_char_ids, cp_sequence_lengths, cp_word_lengths,
                   padded_cn_word_ids, padded_cn_char_ids, cn_sequence_lengths, cn_word_lengths)

    def _make_feed_dict(self, data, keep_prob, is_training):
        return {self.q_word_ids: data[0],
                self.q_char_ids: data[1],
                self.q_sequence_lengths: data[2],
                self.q_word_lengths: data[3],
                self.cp_word_ids: data[4],
                self.cp_char_ids: data[5],
                self.cp_sequence_lengths: data[6],
                self.cp_word_lengths: data[7],
                self.cn_word_ids: data[8],
                self.cn_char_ids: data[9],
                self.cn_sequence_lengths: data[10],
                self.cn_word_lengths: data[11],
                self.keep_op: keep_prob,
                self.is_training: is_training}

    def get_scores(self, question, contexts):
        scores = []

        num_batch = len(contexts) // self.batch_size + 1
        start = 0
        idx = 0
        while idx < num_batch:
            context_batch = contexts[start:start + self.batch_size]
            if not context_batch:
                break

            q_char_ids, q_word_ids = zip(*[zip(*zip(*x)) for x in [question] * len(context_batch)])
            padded_q_word_ids, q_sequence_lengths = pad_sequences(q_word_ids, pad_tok=0)
            padded_q_char_ids, q_word_lengths = pad_sequences(q_char_ids, pad_tok=0, nlevels=2)

            c_char_ids, c_word_ids = zip(*[zip(*zip(*x)) for x in context_batch])
            padded_c_word_ids, c_sequence_lengths = pad_sequences(c_word_ids, pad_tok=0)
            padded_c_char_ids, c_word_lengths = pad_sequences(c_char_ids, pad_tok=0, nlevels=2)

            start += self.batch_size
            idx += 1

            feed_dict = {self.cp_char_ids: padded_c_char_ids,
                         self.cp_word_ids: padded_c_word_ids,
                         self.cp_sequence_lengths: c_sequence_lengths,
                         self.cp_word_lengths: c_word_lengths,
                         self.q_word_ids: padded_q_word_ids,
                         self.q_char_ids: padded_q_char_ids,
                         self.q_sequence_lengths: q_sequence_lengths,
                         self.q_word_lengths: q_word_lengths,
                         self.keep_op: 1.0,
                         self.is_training: False}
            scores.extend(self.sess.run(self.p_score, feed_dict=feed_dict))
        return np.array(scores)


class NoAttendNeuralScoreRanker(PairwiseModel, NoAttention, NeuralScore):
    def __init__(self, model_name, embeddings, num_chars, batch_size, early_stopping, k_neg, fast_sample):
        NoAttention.__init__(self)
        NeuralScore.__init__(self)
        PairwiseModel.__init__(self, model_name, embeddings, num_chars, batch_size, early_stopping, k_neg, fast_sample)

    def _add_model_op(self):
        NoAttention._add_model_op(self)
        NeuralScore._add_model_op(self)
        PairwiseModel._add_model_op(self)

        with tf.variable_scope("question_embeddings"), tf.device("/gpu:0"):
            q_char_output = self.embed_char(self.q_char_ids, self.q_word_lengths)
            _, q_lstm_output = self.contextual_lstm(self.q_word_embeddings, q_char_output, self.q_sequence_lengths)

            dense = tf.layers.dense(q_lstm_output, self.lstm_units, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            dense = tf.nn.dropout(dense, keep_prob=self.keep_op)
            self.q_dense = tf.layers.dense(dense, self.embed_size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name="question_embeddings")

        with tf.variable_scope("neg_context_embeddings"), tf.device("/gpu:1"):
            cn_char_output = self.embed_char(self.cn_char_ids, self.cn_word_lengths)
            _, cn_lstm_output = self.contextual_lstm(self.cn_word_embeddings, cn_char_output, self.cn_sequence_lengths)
            self.cn_dense = self.context_representation(cn_lstm_output)

        with tf.variable_scope("pos_context_embeddings"), tf.device("/gpu:0"):
            cp_char_output = self.embed_char(self.cp_char_ids, self.cp_word_lengths)
            _, cp_lstm_output = self.contextual_lstm(self.cp_word_embeddings, cp_char_output, self.cp_sequence_lengths)
            self.cp_dense = self.context_representation(cp_lstm_output)

        with tf.variable_scope("scoring_function"), tf.device("/gpu:1"):
            self.n_score = self.score("neg")
            self.p_score = self.score("pos")


class FixedAttendNeuralScoreRanker(PairwiseModel, FixAttention, NeuralScore):
    def __init__(self, model_name, embeddings, num_chars, batch_size, early_stopping, k_neg, fast_sample):
        FixAttention.__init__(self)
        NeuralScore.__init__(self)
        PairwiseModel.__init__(self, model_name, embeddings, num_chars, batch_size, early_stopping, k_neg, fast_sample)

    def _add_model_op(self):
        FixAttention._add_model_op(self)
        NeuralScore._add_model_op(self)
        PairwiseModel._add_model_op(self)

        with tf.variable_scope("question_embeddings"), tf.device("/gpu:0"):
            q_char_output = self.embed_char(self.q_char_ids, self.q_word_lengths)
            _, q_lstm_output = self.contextual_lstm(self.q_word_embeddings, q_char_output, self.q_sequence_lengths)

            dense = tf.layers.dense(q_lstm_output, self.lstm_units, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            dense = tf.nn.dropout(dense, keep_prob=self.keep_op)
            self.q_dense = tf.layers.dense(dense, self.embed_size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name="question_embeddings")

        with tf.variable_scope("neg_context_embeddings"), tf.device("/gpu:1"):
            cn_char_output = self.embed_char(self.cn_char_ids, self.cn_word_lengths)
            self.cn_embeddings, _ = self.contextual_lstm(self.cn_word_embeddings, cn_char_output,
                                                         self.cn_sequence_lengths)
            self.cn_dense = self.context_representation(self.cn_embeddings, self.cn_sequence_lengths)

        with tf.variable_scope("pos_context_embeddings"), tf.device("/gpu:0"):
            cp_char_output = self.embed_char(self.cp_char_ids, self.cp_word_lengths)
            self.cp_embeddings, _ = self.contextual_lstm(self.cp_word_embeddings, cp_char_output,
                                                         self.cp_sequence_lengths)
            self.cp_dense = self.context_representation(self.cp_embeddings, self.cp_sequence_lengths)

        with tf.variable_scope("scoring_function"), tf.device("/gpu:1"):
            self.n_score = self.score("neg")
            self.p_score = self.score("pos")

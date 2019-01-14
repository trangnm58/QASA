import tensorflow as tf

from constants import PARAMS


def mask_softmax(logits, seq_lens, seq_len, dim=-1, name=""):
    mask = tf.sequence_mask(seq_lens, seq_len, dtype=tf.float32)
    true_logits = tf.multiply(tf.exp(logits), mask)
    return tf.divide(true_logits, tf.reduce_sum(true_logits, dim, keep_dims=True), name=name)


class CharLSTM:
    def __init__(self):
        self.char_lstm_units = PARAMS['char_lstm_units']
        self.char_dim = PARAMS['char_dim']
        self.num_chars = None
        self.embed_char = self._lstm_char

    def _add_model_op(self):
        with tf.variable_scope("char-lstm"):
            self.char_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.char_lstm_units)
            self.char_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.char_lstm_units)

            self._char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                                                    shape=[self.num_chars, self.char_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())

    def _lstm_char(self, char_ids, word_lengths):
        char_embeddings = tf.nn.embedding_lookup(self._char_embeddings, char_ids, name="char_embeddings")
        # put the time dimension on axis=1
        s = tf.shape(char_embeddings)
        r_char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.char_dim])
        word_lengths = tf.reshape(word_lengths, shape=[-1])

        # bi lstm on chars
        _, ((_, char_fw), (_, char_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.char_cell_fw,
                                                                          cell_bw=self.char_cell_bw,
                                                                          inputs=r_char_embeddings,
                                                                          sequence_length=word_lengths,
                                                                          dtype=tf.float32)
        char_output = tf.concat([char_fw, char_bw], axis=-1)
        # shape = (batch size, max sentence length, char hidden size)
        char_output = tf.reshape(char_output, shape=[-1, s[1], 2 * self.char_lstm_units])
        return char_output


class ContextualLSTM:
    def __init__(self):
        self.lstm_units = PARAMS['lstm_units']

    def _add_model_op(self):
        with tf.variable_scope("contextual-lstm"):
            self.cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_units)
            self.cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_units)

    def contextual_lstm(self, word_embeddings, char_embeddings, seq_lens):
        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

        (states_fw, states_bw), ((_, last_state_fw), (_, last_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            inputs=embeddings,
            sequence_length=seq_lens,
            dtype=tf.float32)

        H = tf.concat([states_fw, states_bw], axis=2)
        lstm_output = tf.concat([last_state_fw, last_state_bw], axis=-1)
        return H, lstm_output


class NoAttention:
    def __init__(self):
        self.embed_size = PARAMS['embed_size']
        self.lstm_units = PARAMS['lstm_units']

        # Required Tensors and Ops
        self.keep_op = None

    def _add_model_op(self):
        with tf.variable_scope("context-representation"):
            self.c_W1 = tf.get_variable("c_W1", shape=[2 * self.lstm_units, self.lstm_units],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.c_b1 = tf.get_variable("c_b1", shape=[self.lstm_units], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
            self.c_W2 = tf.get_variable("c_W2", shape=[self.lstm_units, self.embed_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.c_b2 = tf.get_variable("c_b2", shape=[self.embed_size], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

    def context_representation(self, c_lstm_output):
        with tf.name_scope("dense"):
            c_dense = tf.nn.relu(tf.nn.xw_plus_b(c_lstm_output, self.c_W1, self.c_b1))
            c_dense = tf.nn.dropout(c_dense, keep_prob=self.keep_op)
            c_dense = tf.nn.xw_plus_b(c_dense, self.c_W2, self.c_b2)
        return c_dense


class CustomAttention:
    def __init__(self):
        self.attend_W1_size = PARAMS['attend_W1']
        self.embed_size = PARAMS['embed_size']
        self.lstm_units = PARAMS['lstm_units']

        # Required Tensors and Ops
        self.q_dense = None
        self.keep_op = None

        # New define Tensors
        self.A = None

    def _add_model_op(self):
        with tf.variable_scope("context-representation"):
            self.c_attend_W1 = tf.get_variable(name="attend_W1", dtype=tf.float32,
                                               shape=[2 * self.lstm_units + self.embed_size,
                                                      self.attend_W1_size],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.c_attend_b1 = tf.get_variable("c_attend_b1", shape=[self.attend_W1_size], dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

            self.c_attend_W2 = tf.get_variable(name="attend_W2", dtype=tf.float32,
                                               shape=[self.attend_W1_size, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.c_attend_b2 = tf.get_variable("c_attend_b2", shape=[1], dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

            self.c_W1 = tf.get_variable("c_W1", shape=[2 * self.lstm_units, self.embed_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.c_b1 = tf.get_variable("c_b1", shape=[self.embed_size], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

    def context_representation(self, H, seq_lens):
        seq_len = tf.shape(H)[1]

        q_dense = tf.tile(self.q_dense, [1, seq_len])
        q_dense = tf.reshape(q_dense, [-1, seq_len, self.embed_size])
        H_q = tf.concat([H, q_dense], axis=2)
        H_q_reshape = tf.reshape(H_q, [-1, 2 * self.lstm_units + self.embed_size])

        with tf.name_scope("self-attention"):
            H_q_s1 = tf.nn.relu(tf.nn.xw_plus_b(H_q_reshape, self.c_attend_W1, self.c_attend_b1))
            H_q_s1 = tf.nn.dropout(H_q_s1, self.keep_op)
            H_q_s2 = tf.nn.xw_plus_b(H_q_s1, self.c_attend_W2, self.c_attend_b2)
            H_q_s2 = tf.reshape(H_q_s2, shape=[-1, seq_len])
            self.A = mask_softmax(H_q_s2, seq_lens, seq_len, name="attention", dim=1)
            A = tf.reshape(self.A, shape=[-1, seq_len, 1])
            context_vector = tf.reduce_sum(tf.multiply(A, H), axis=1, name="context_vector")  # bs, 2*units

        with tf.name_scope("dense"):
            c_dense = tf.nn.xw_plus_b(context_vector, self.c_W1, self.c_b1)
        return c_dense


class FixAttention(CustomAttention):
    def __init__(self):
        CustomAttention.__init__(self)

    def _add_model_op(self):
        CustomAttention._add_model_op(self)

    def context_representation(self, H, seq_lens):
        seq_len = tf.shape(H)[1]

        q_dense = tf.stop_gradient(self.q_dense, name="q_dense_fixed")
        q_dense = tf.tile(q_dense, [1, seq_len])
        q_dense = tf.reshape(q_dense, [-1, seq_len, self.embed_size])
        H_q = tf.concat([H, q_dense], axis=2)
        H_q_reshape = tf.reshape(H_q, [-1, 2 * self.lstm_units + self.embed_size])

        with tf.name_scope("self-attention"):
            H_q_s1 = tf.nn.relu(tf.nn.xw_plus_b(H_q_reshape, self.c_attend_W1, self.c_attend_b1))
            H_q_s1 = tf.nn.dropout(H_q_s1, self.keep_op)
            H_q_s2 = tf.nn.xw_plus_b(H_q_s1, self.c_attend_W2, self.c_attend_b2)
            H_q_s2 = tf.nn.dropout(H_q_s2, self.keep_op)
            H_q_s2 = tf.reshape(H_q_s2, shape=[-1, seq_len])
            self.A = mask_softmax(H_q_s2, seq_lens, seq_len, name="attention", dim=1)
            A = tf.reshape(self.A, shape=[-1, seq_len, 1])
            context_vector = tf.reduce_sum(tf.multiply(A, H), axis=1, name="context_vector")  # bs, 2*units

        with tf.name_scope("dense"):
            c_dense = tf.nn.xw_plus_b(context_vector, self.c_W1, self.c_b1)
        return c_dense


class NeuralScore:
    def __init__(self):
        self.embed_size = PARAMS['embed_size']
        self.scoring_W1_size = PARAMS['scoring_W1']

        # Required Tensors and Ops
        self.q_dense = None
        self.cn_dense = None
        self.cp_dense = None
        self.keep_op = None

    def _add_model_op(self):
        self.s_W1 = tf.get_variable(name="scoring_W1", dtype=tf.float32,
                                    shape=[4 * self.embed_size, self.scoring_W1_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.s_b1 = tf.get_variable(name="scoring_b1", dtype=tf.float32,
                                    shape=[self.scoring_W1_size], initializer=tf.zeros_initializer())
        self.s_W2 = tf.get_variable(name="scoring_W2", dtype=tf.float32,
                                    shape=[self.scoring_W1_size, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.s_b2 = tf.get_variable(name="scoring_b2", dtype=tf.float32,
                                    shape=[1], initializer=tf.zeros_initializer())

    def score(self, branch):
        if branch == "neg":
            c_dense = self.cn_dense
        else:
            c_dense = self.cp_dense

        features = tf.concat([self.q_dense, c_dense, tf.abs(self.q_dense - c_dense),
                              tf.multiply(self.q_dense, c_dense)], axis=-1)
        s = tf.nn.relu(tf.nn.xw_plus_b(features, self.s_W1, self.s_b1))
        s = tf.nn.dropout(s, self.keep_op)
        s = tf.nn.xw_plus_b(s, self.s_W2, self.s_b2)
        s = tf.reshape(s, shape=[-1])
        return s

import os
import argparse
import numpy as np
import tensorflow as tf
from keras.layers import Input

from dataset import QuasarTDataset, get_trimmed_embeddings, pad_sequences
from constants import DATA, TRAINED_MODELS, PARAMS
from utils import Timer


class PairwiseModel:
    def __init__(self, model_name, embeddings, batch_size, early_stopping, k_neg):
        self.model_name = TRAINED_MODELS + model_name + "/"
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        self.epochs = None
        self.train_examples = None
        self.dev_examples = None
        self.data = None
        self.k_neg = k_neg
        self.keep_prob = None

        self.word_dim = PARAMS['word_dim']
        self.relation_size_1 = 300
        self.relation_size_2 = 5
        self.keep_prob = PARAMS['keep_prob']

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        self.sess = tf.InteractiveSession(config=config)

    def _add_placeholders(self):
        self.cp_word_ids = Input(name="cp_word_ids", shape=[None], dtype='int32')
        self.cp_sequence_lengths = Input(name='cp_sequence_lengths', batch_shape=[None], dtype='int32')

        self.cn_word_ids = Input(name="cn_word_ids", shape=[None], dtype='int32')
        self.cn_sequence_lengths = Input(name='cn_sequence_lengths', batch_shape=[None], dtype='int32')

        self.q_word_ids = Input(name="q_word_ids", shape=[None], dtype='int32')
        self.q_sequence_lengths = Input(name='q_sequence_lengths', batch_shape=[None], dtype='int32')

        self.keep_op = tf.placeholder(dtype=tf.float32, shape=[], name="keep_op")
        self.is_training = tf.placeholder(tf.bool, name='phase')

    def _add_model_op(self):
        with tf.variable_scope("shared_cells"):
            self._word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                                trainable=False)

            # Weights for relation scoring function
            self.r_W1 = tf.get_variable(name="relation_W1", dtype=tf.float32,
                                        shape=[self.word_dim*2, self.relation_size_1],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.r_b1 = tf.get_variable(name="relation_b1", dtype=tf.float32,
                                        shape=[self.relation_size_1], initializer=tf.zeros_initializer())

            self.r_W2 = tf.get_variable(name="relation_W2", dtype=tf.float32,
                                        shape=[self.relation_size_1, self.relation_size_1],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.r_b2 = tf.get_variable(name="relation_b2", dtype=tf.float32,
                                        shape=[self.relation_size_1], initializer=tf.zeros_initializer())

            self.r_W3 = tf.get_variable(name="relation_W3", dtype=tf.float32,
                                        shape=[self.relation_size_1, self.relation_size_2],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.r_b3 = tf.get_variable(name="relation_b3", dtype=tf.float32,
                                        shape=[self.relation_size_2], initializer=tf.zeros_initializer())

            self.r_W4 = tf.get_variable(name="relation_W4", dtype=tf.float32,
                                        shape=[self.relation_size_2, self.relation_size_2],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.r_b4 = tf.get_variable(name="relation_b4", dtype=tf.float32,
                                        shape=[self.relation_size_2], initializer=tf.zeros_initializer())

            self.r_W5 = tf.get_variable(name="relation_W5", dtype=tf.float32,
                                        shape=[self.relation_size_2, self.relation_size_2],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.r_b5 = tf.get_variable(name="relation_b5", dtype=tf.float32,
                                        shape=[self.relation_size_2], initializer=tf.zeros_initializer())

            self.r_W6 = tf.get_variable(name="relation_W6", dtype=tf.float32,
                                        shape=[self.relation_size_2, 1],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.r_b6 = tf.get_variable(name="relation_b6", dtype=tf.float32,
                                        shape=[1], initializer=tf.zeros_initializer())

        with tf.variable_scope("question_embeddings"), tf.device("/gpu:0"):
            self.q_word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.q_word_ids,
                                                            name="q_word_embeddings")

        with tf.variable_scope("neg_context_embeddings"), tf.device("/gpu:1"):
            self.cn_word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.cn_word_ids,
                                                             name="cn_word_embeddings")

        with tf.variable_scope("pos_context_embeddings"), tf.device("/gpu:0"):
            self.cp_word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.cp_word_ids,
                                                             name="cp_word_embeddings")

        def relation_score(c_embeddings):
            q_seq_len = tf.shape(self.q_word_embeddings)[1]
            c_seq_len = tf.shape(c_embeddings)[1]
            question_embeddings = tf.tile(self.q_word_embeddings, [1, 1, c_seq_len])
            question_embeddings = tf.reshape(question_embeddings, shape=[-1, q_seq_len, c_seq_len, self.word_dim])

            context_embeddings = tf.tile(c_embeddings, [1, q_seq_len, 1])
            context_embeddings = tf.reshape(context_embeddings, shape=[-1, q_seq_len, c_seq_len, self.word_dim])

            combined_embeddings = tf.concat([question_embeddings, context_embeddings], axis=-1)
            combined_embeddings = tf.reshape(combined_embeddings, shape=[-1, self.word_dim*2])

            r = tf.nn.relu(tf.nn.xw_plus_b(combined_embeddings, self.r_W1, self.r_b1))
            r = tf.nn.dropout(r, self.keep_op)
            r = tf.nn.relu(tf.nn.xw_plus_b(r, self.r_W2, self.r_b2))
            r = tf.nn.dropout(r, self.keep_op)
            r = tf.nn.relu(tf.nn.xw_plus_b(r, self.r_W3, self.r_b3))

            r = tf.reshape(r, shape=[-1, q_seq_len * c_seq_len, self.relation_size_2])
            r = tf.reduce_sum(r, axis=1, keep_dims=False)

            r = tf.nn.relu(tf.nn.xw_plus_b(r, self.r_W4, self.r_b4))
            r = tf.nn.relu(tf.nn.xw_plus_b(r, self.r_W5, self.r_b5))
            r = tf.nn.xw_plus_b(r, self.r_W6, self.r_b6)
            r = tf.reshape(r, shape=[-1])
            return r

        with tf.variable_scope("scoring_function"), tf.device("/gpu:1"):
            with tf.variable_scope("negative"):
                self.n_score = relation_score(self.cn_word_embeddings)
            with tf.variable_scope("positive"):
                self.p_score = relation_score(self.cp_word_embeddings)

    def _add_train_op(self):
        margin = 1.0

        self.loss_op = tf.reduce_mean(tf.maximum(0., margin - self.p_score + self.n_score))
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss_op)

    def build(self):
        timer = Timer()
        timer.start("Building model")

        self._add_placeholders()
        self._add_model_op()
        self._add_train_op()

        timer.stop()

    def _next_batch(self, data, num_batch):
        yield None

    def _make_feed_dict(self, data, keep_prob, is_training):
        return []

    def _dev_acc(self, top_k=1):
        return 0

    def _train(self, epochs, is_continue, patience=4):
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        saver = tf.train.Saver(max_to_keep=1)
        if is_continue:
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_name))
        else:
            self.sess.run(tf.global_variables_initializer())

        best_acc = 0
        nepoch_noimp = 0
        t = Timer()
        for e in range(epochs):
            t.start("Epoch {}".format(e + 1))
            total_train_loss = 0
            c = 0
            t.start("Create training examples")
            train_examples = self.data.create_train_examples_all_pos_rand_neg(self)
            print("Number of training examples in {}: {}".format(e + 1, len(train_examples)))
            t.stop()
            num_batch_train = len(train_examples) // self.batch_size + 1
            display_step = num_batch_train // 4

            for idx, batch in enumerate(self._next_batch(data=train_examples, num_batch=num_batch_train)):
                feed_dict = self._make_feed_dict(batch, self.keep_prob, True)
                _, train_loss = self.sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)
                total_train_loss += train_loss
                c += 1
                if idx % display_step == 0:
                    print("Iter {} - Loss: {}".format(idx, total_train_loss / c))

            if self.early_stopping:
                dev_acc = self._dev_acc(top_k=1)
                print("Dev accuracy (top {}): {}".format(1, dev_acc))
                if dev_acc > best_acc:
                    saver.save(self.sess, self.model_name + "model")
                    print('Saved the model at epoch {}'.format(e + 1))
                    best_acc = dev_acc
                    nepoch_noimp = 0
                else:
                    nepoch_noimp += 1
                    print("Number of epochs with no improvement: {}".format(nepoch_noimp))
                    if nepoch_noimp >= patience:
                        break
            t.stop()
        if not self.early_stopping:
            saver.save(self.sess, self.model_name + "model")
            print('Saved the model')

        self.sess.close()

    def load_data(self, dataset):
        timer = Timer()
        timer.start("Loading data")

        self.data = dataset
        self.train_examples = self.data.train_examples

        if self.early_stopping:
            self.dev_examples = self.data.dev_examples
            print("Number of validating mentions: {}".format(len(self.dev_examples)))

        print("Number of training mentions: {}".format(len(self.train_examples)))

        timer.stop()

    def run_train(self, epochs, is_continue=False, patience=4):
        self.epochs = epochs
        timer = Timer()
        timer.start("Training model")
        self._train(epochs, is_continue=is_continue, patience=patience)
        timer.stop()

    def get_scores(self, question, contexts):
        return []


class QuasarModel(PairwiseModel):
    def __init__(self, model_name, embeddings, batch_size=64, early_stopping=False, k_neg=50):
        PairwiseModel.__init__(self, model_name, embeddings, batch_size, early_stopping, k_neg)

    def _add_model_op(self):
        PairwiseModel._add_model_op(self)

    def _dev_acc(self, top_k=1):
        data = self.data.dev_data

        count_true = 0
        indexes = list(data.keys())
        count_total = len(indexes)

        for i in indexes:
            question = data[i]['q']
            pos_cs = data[i]['pos_contexts']
            neg_cs = data[i]['neg_contexts']

            pivot = len(pos_cs)  # index < pivot is positive context
            all_cs = pos_cs + neg_cs

            all_context_examples = [self.data.process_sent(c, update_max=False) for c in all_cs]
            question_example = self.data.process_sent(question, update_max=False)
            scores = self.get_scores(question_example, all_context_examples)
            c_max = scores.argsort()[::-1][:top_k]
            if min(c_max) < pivot:
                count_true += 1

        return count_true / count_total

    def evaluate(self, data, top_k=(1, 3, 5)):
        timer = Timer()
        timer.start("Evaluating on a given dataset")

        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_name))

        count_true = [0] * len(top_k)
        indexes = list(data.keys())
        count_total = len(indexes)
        for i in indexes:
            question = data[i]['q']
            pos_cs = data[i]['pos_contexts']
            neg_cs = data[i]['neg_contexts']

            pivot = len(pos_cs)  # index < pivot is positive context
            all_cs = pos_cs + neg_cs

            all_context_examples = [self.data.process_sent(c, update_max=False) for c in all_cs]
            question_example = self.data.process_sent(question, update_max=False)
            scores = self.get_scores(question_example, all_context_examples)

            for j in range(len(top_k)):
                c_max = scores.argsort()[::-1][:top_k[j]]
                if min(c_max) < pivot:
                    count_true[j] += 1

        for i in range(len(top_k)):
            print("Top {}:".format(top_k[i]))
            print("- Accuracy: {}".format(count_true[i] / count_total))
            print("- Total: {} - Correct: {}".format(count_total, count_true[i]))
        self.sess.close()
        timer.stop()

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0

        while idx < num_batch:
            example_batch = data[start:start + self.batch_size]
            if not example_batch:
                break

            q_batch, cp_batch, cn_batch = zip(*example_batch)

            _, q_word_ids = zip(*[zip(*zip(*x)) for x in q_batch])
            padded_q_word_ids, q_sequence_lengths = pad_sequences(q_word_ids, pad_tok=0)

            _, cp_word_ids = zip(*[zip(*zip(*x)) for x in cp_batch])
            padded_cp_word_ids, cp_sequence_lengths = pad_sequences(cp_word_ids, pad_tok=0)

            _, cn_word_ids = zip(*[zip(*zip(*x)) for x in cn_batch])
            padded_cn_word_ids, cn_sequence_lengths = pad_sequences(cn_word_ids, pad_tok=0)

            start += self.batch_size
            idx += 1
            yield (padded_q_word_ids, q_sequence_lengths,
                   padded_cp_word_ids, cp_sequence_lengths,
                   padded_cn_word_ids, cn_sequence_lengths)

    def _make_feed_dict(self, data, keep_prob, is_training):
        return {self.q_word_ids: data[0],
                self.q_sequence_lengths: data[1],
                self.cp_word_ids: data[2],
                self.cp_sequence_lengths: data[3],
                self.cn_word_ids: data[4],
                self.cn_sequence_lengths: data[5],
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

            _, q_word_ids = zip(*[zip(*zip(*x)) for x in [question] * len(context_batch)])
            padded_q_word_ids, q_sequence_lengths = pad_sequences(q_word_ids, pad_tok=0)

            _, c_word_ids = zip(*[zip(*zip(*x)) for x in context_batch])
            padded_c_word_ids, c_sequence_lengths = pad_sequences(c_word_ids, pad_tok=0)

            start += self.batch_size
            idx += 1

            feed_dict = {self.cp_word_ids: padded_c_word_ids,
                         self.cp_sequence_lengths: c_sequence_lengths,
                         self.q_word_ids: padded_q_word_ids,
                         self.q_sequence_lengths: q_sequence_lengths,
                         self.keep_op: 1.0,
                         self.is_training: False}
            scores.extend(self.sess.run(self.p_score, feed_dict=feed_dict))
        return np.array(scores)


def main(model_name, dataset, train_set, is_evaluate, early_stopping, epoch, k_neg, is_continue):
    train = QuasarTDataset(dataset)
    train.load_data_pairwise(DATA + "{}/{}".format(dataset, train_set))

    embeddings = get_trimmed_embeddings(DATA + "{}/embedding_data.npz".format(dataset))

    model = QuasarModel(model_name=model_name, embeddings=embeddings,
                        batch_size=32, early_stopping=early_stopping, k_neg=k_neg)
    model.load_data(train)
    model.build()

    if is_evaluate:
        model.evaluate(train.test_data)
    elif early_stopping:
        model.run_train(epoch, is_continue, patience=5)
    else:
        model.run_train(epoch, is_continue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train new model.')
    parser.add_argument('model', help="the name of the model")
    parser.add_argument('dataset', help="the name of the dataset that the model will be trained on, e.g: quasart_glove")
    parser.add_argument('train_set', help="e.g: train_pairwise.pickle")

    # optional
    parser.add_argument("-eval", "--evaluate", help="evaluate existing model", action="store_true")
    parser.add_argument("-es", "--early_stopping", help="use early stopping", action="store_true")
    parser.add_argument("-e", "--epoch", help="number of epochs to train or maximum epoch when early stopping",
                        type=int, default=10)
    parser.add_argument("-k", "--k_neg", help="number of negative examples to be sampled", type=int, default=50)
    parser.add_argument("-c", "--is_continue", help="continue to train or not", action="store_true")
    args = parser.parse_args()

    main(args.model, args.dataset, args.train_set, args.evaluate,
         args.early_stopping, args.epoch, args.k_neg, args.is_continue)

import os
import tensorflow as tf

from constants import TRAINED_MODELS
from utils import Timer


class DeepModel:
    def __init__(self, model_name, batch_size, early_stopping, fast_sample):
        self.model_name = TRAINED_MODELS + model_name + "/"
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.fast_sample = fast_sample

        self.epochs = None
        self.train_examples = None
        self.dev_examples = None
        self.data = None
        self.keep_prob = None

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        self.sess = tf.InteractiveSession(config=config)

    def _add_placeholders(self):
        pass

    def _add_model_op(self):
        pass

    def _add_train_op(self):
        self.loss_op = None
        self.train_op = None

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
        return {}

    def _dev_acc(self):
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
            if self.fast_sample or nepoch_noimp == 0:
                train_examples = self.data.create_train_examples_all_pos_rand_neg(self)
            else:
                train_examples = self.data.create_train_examples_all_pos_top_neg(self)
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
                dev_acc = self._dev_acc()
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

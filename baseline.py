import argparse
import tensorflow as tf

from models.pairwise_model import NoAttendNeuralScoreRanker
from utils import Timer
from dataset import QuasarTDataset, get_trimmed_embeddings
from constants import DATA


class QuasarModel(NoAttendNeuralScoreRanker):
    def __init__(self, model_name, embeddings, num_chars, batch_size=64,
                 early_stopping=False, k_neg=50, fast_sample=False):
        super().__init__(model_name, embeddings, num_chars, batch_size, early_stopping, k_neg, fast_sample)

    def _add_model_op(self):
        super()._add_model_op()

    def _dev_acc(self, top_k=1):
        data = self.data.test_data

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


def main(model_name, dataset, train_set, is_evaluate, early_stopping, epoch, k_neg, is_continue):
    train = QuasarTDataset(dataset)

    train.load_data_pairwise(DATA + "{}/{}".format(dataset, train_set))
    num_chars = train.get_num_chars()

    embeddings = get_trimmed_embeddings(DATA + "{}/embedding_data.npz".format(dataset))

    model = QuasarModel(model_name=model_name, embeddings=embeddings, num_chars=num_chars,
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
    parser.add_argument('dataset', help="the name of the dataset that the model will be trained on, e.g: quasart")
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

import argparse
import pickle

from pre_process import PreProcess
from constants import DATA, CONVERTED_DATA
from utils import Timer


class QuasarTConverter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_file = "train_raw.pickle"
        self.dev_file = "dev_raw.pickle"
        self.test_file = "test_raw.pickle"

    def convert_data(self):
        train_raw = self._get_data(self.train_file)
        dev_raw = self._get_data(self.dev_file)
        test_raw = self._get_data(self.test_file)

        train_data = self._convert_data(train_raw)
        dev_data = self._convert_data(dev_raw)
        test_data = self._convert_data(test_raw)

        with open(DATA + self.dataset + "/" + CONVERTED_DATA, 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dev_data, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    def _get_data(self, data_file):
        with open(DATA + "{}/{}".format(self.dataset, data_file), 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def _get_pos_neg_contexts(answer, contexts):
        cs = [c.lower() for c in contexts]
        a = answer.lower()
        pos_cs = []
        neg_cs = []
        for c in cs:
            if a in c:
                pos_cs.append(c)
            else:
                neg_cs.append(c)
        return pos_cs, neg_cs

    def _convert_data(self, data_obj):
        pre_process = PreProcess()

        data = {}
        indexes = list(data_obj.keys())
        count_true = 0
        for idx in indexes:
            cs = data_obj[idx]['contexts']
            q = data_obj[idx]['question']
            a = data_obj[idx]['answer']

            pos_cs, neg_cs = self._get_pos_neg_contexts(a, cs)
            if pos_cs:
                count_true += 1
            pos_contexts = [" ".join(pre_process.process(c, remove_stop_words=False,
                                                         simple_tokenize=True)) for c in pos_cs]
            neg_contexts = [" ".join(pre_process.process(c, remove_stop_words=False,
                                                         simple_tokenize=True)) for c in neg_cs]

            question = " ".join(pre_process.process(q, remove_stop_words=False,
                                                    simple_tokenize=True))

            data[idx] = {
                'pos_contexts': pos_contexts,
                'neg_contexts': neg_contexts,
                'q': question
            }
        print(count_true)
        return data


def main(dataset):
    c = QuasarTConverter(dataset)
    t = Timer()
    t.start("Converting data", verbal=True)

    c.convert_data()
    t.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert data from a particular format into uniform input.')
    parser.add_argument('dataset', help="the name of the raw dataset to be converted, e.g: quasart")

    args = parser.parse_args()

    main(args.dataset)

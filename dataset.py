import pickle
import numpy as np
import argparse
import random
from sklearn.utils import shuffle, resample

from utils import Timer
from constants import DATA, UNK, CONVERTED_DATA


def get_trimmed_embeddings(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename, encoding='utf8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def pad_sequences(sequences, pad_tok, nlevels=1, fixed_word_len=None, fixed_sent_len=None):
    def _pad_sequences(seqs, tok, max_len):
        seq_padded, seq_len = [], []

        for seq in seqs:
            seq = list(seq)
            seq_ = seq[:max_len] + [tok] * max(max_len - len(seq), 0)
            seq_padded += [seq_]
            seq_len += [min(len(seq), max_len)]

        return seq_padded, seq_len

    if nlevels == 1:
        if fixed_sent_len:
            max_length = fixed_sent_len
        else:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    else:
        if fixed_word_len:
            max_length_word = fixed_word_len
        else:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset

        self.train_examples = None
        self.dev_examples = None

        self._load_vocabs()
        self.max_word_length = 0
        self.max_sent_length = 0
        self.max_negative_set = 0

    def _load_data(self):
        return None, None

    def create_pairwise(self):
        return None, None

    def save_pairwise(self, file_name, data):
        pass

    def load_data_pairwise(self, file_name):
        pass

    def get_num_chars(self):
        path = DATA + "{}/all_chars.txt".format(self.dataset)
        with open(path, 'r', encoding='utf8') as f:
            d = f.readlines()
        return len(d)

    def create_train_examples_all_pos_rand_neg(self, model):
        train_examples = []
        for e in self.train_examples:
            if model.k_neg >= len(e[2]):
                neg_samples = e[2]
            else:
                neg_samples = resample(e[2], replace=False, n_samples=model.k_neg)

            for cp in e[1]:
                for cn in neg_samples:
                    train_examples.append([e[0], cp, cn])
        return shuffle(train_examples)

    def create_train_examples_all_pos_top_neg(self, model):
        k = 20
        train_examples = []
        for e in self.train_examples:
            scores = model.get_scores(e[0], e[2])
            top_neg = scores.argsort()[::-1][:k]

            for c in e[1]:
                neg = random.choice(top_neg)
                train_examples.append([e[0], c, e[2][neg]])

        return shuffle(train_examples)

    def process_sent(self, text, update_max=True):
        words = text.split()

        char_ids = []
        word_ids = []
        for w in words:
            c_ids = []
            # get chars of word
            for char in w:
                # ignore chars out of vocabulary
                if char in self.vocab_chars:
                    c_ids.append(self.vocab_chars[char])

            char_ids.append(c_ids)
            if update_max and len(c_ids) > self.max_word_length:
                self.max_word_length = len(char_ids)

            if w in self.vocab_words:
                word_ids.append(self.vocab_words[w])
            else:
                word_ids.append(self.vocab_words[UNK])

        if update_max and len(word_ids) > self.max_sent_length:
            self.max_sent_length = len(word_ids)

        # return tuple char ids, word id
        return char_ids, word_ids

    def _load_vocabs(self):
        self.vocab_words = load_vocab(DATA + "{}/all_words.txt".format(self.dataset))
        self.vocab_chars = load_vocab(DATA + "{}/all_chars.txt".format(self.dataset))


class QuasarTDataset(Dataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.train_data, self.dev_data, self.test_data = self._load_data()

    def create_pairwise(self):
        t = Timer()
        t.start("Creating pair-wise dataset")
        examples = []
        indexes = list(self.train_data.keys())

        for idx in indexes:
            pos_contexts = self.train_data[idx]['pos_contexts']
            neg_contexts = self.train_data[idx]['neg_contexts']
            question = self.train_data[idx]['q']

            if not (pos_contexts and neg_contexts):
                continue

            pos_examples = []
            for c in pos_contexts:
                pos_examples.append(self.process_sent(c))

            neg_examples = []
            for c in neg_contexts:
                neg_examples.append(self.process_sent(c))
            if len(neg_examples) > self.max_negative_set:
                self.max_negative_set = len(neg_examples)

            examples.append([self.process_sent(question), pos_examples, neg_examples])

        t.stop()
        return shuffle(examples)

    def save_pairwise(self, file_name, data):
        t = Timer()
        t.start("Saving pair-wise dataset", verbal=True)
        print("Max sentence length: {}\nMax word length: {}".format(self.max_sent_length, self.max_word_length))
        print("Max negative set size: {}".format(self.max_negative_set))
        print("Number of training examples: {}".format(len(data)))
        with open(file_name, 'wb') as f:
            pickle.dump((self.max_word_length, self.max_sent_length), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        t.stop()

    def load_data_pairwise(self, file_name):
        with open(file_name, 'rb') as f:
            self.max_word_length, self.max_sent_length = pickle.load(f)
            self.train_examples = pickle.load(f)
            self.dev_examples = []  # dummy

    def get_vocabs(self):
        vocab_word = set()
        vocab_char = set()

        for data in [self.train_data, self.dev_data, self.test_data]:
            for _, v in data.items():
                for c in v['pos_contexts']:
                    context_tokens = c.split()
                    vocab_word.update(context_tokens)
                    for t in context_tokens:
                        vocab_char.update(t)
                for c in v['neg_contexts']:
                    context_tokens = c.split()
                    vocab_word.update(context_tokens)
                    for t in context_tokens:
                        vocab_char.update(t)

                question_tokens = v['q'].split()
                vocab_word.update(question_tokens)
                for t in question_tokens:
                    vocab_char.update(t)

        print("- Done. {} tokens".format(len(vocab_word)))
        return vocab_word, vocab_char

    def _load_data(self):
        with open(DATA + self.dataset + "/" + CONVERTED_DATA, 'rb') as f:
            train_data = pickle.load(f)
            dev_data = pickle.load(f)
            test_data = pickle.load(f)
        return train_data, dev_data, test_data


def main(dataset, data_file):
    d = QuasarTDataset(dataset)

    pairwise_data = d.create_pairwise()
    d.save_pairwise(DATA + "{}/{}.pickle".format(dataset, data_file), pairwise_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build necessary data for model training and evaluating.')
    parser.add_argument('dataset', help="the name of the dataset that the model will be trained on, e.g: quasart")
    parser.add_argument('data_file', help="the name of the dataset saved file")

    args = parser.parse_args()
    main(args.dataset, args.data_file)

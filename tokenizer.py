import collections
from keras.preprocessing.sequence import pad_sequences
import pickle
from tqdm import tqdm
import numpy as np


# 0 is reserved
# make UNK token something
# make vocab size a requirement on creation
class Tokenizer(object):
    def __init__(self):
        self.word_count = collections.Counter()
        self.w2i = {}
        self.i2w = {}
        self.oov_index = None
        self.vocab_size = None
        self.vectors = {}

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    def load(self, path):
        return pickle.load(open(path, 'rb'))

    def train(self, texts, vocab_size):

        if len(self.word_count) != 0:
            raise Exception("To update existing tokenizer with new vocabulary, run update() or update_from_file()")

        # takes a list of strings that are space delimited into tokens
        for sent in texts:
            for w in sent.split():
                self.word_count[w] += 1

        self.vocab_size = vocab_size

        # Easily changed but vocab size is essentially defining your max index
        # 0 is reserved and UNK is reserved so you will get vocab_size-2 to get
        # indices from 0-(vocab_size-1) i.e. vocab_size = 50, we get indices 0-49
        for count, w in enumerate(self.word_count.most_common(self.vocab_size - 2)):
            self.w2i[w[0]] = count + 1
            self.i2w[count + 1] = w[0]
        self.oov_index = min([self.vocab_size - 1, len(self.word_count) + 1])
        self.vocab_size = self.oov_index + 1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    def train_from_file(self, path, vocab_size):
        if len(self.word_count) != 0:
            raise Exception("To update existing tokenizer with new vocabulary, run update() or update_from_file()")

        # This is for our file representation of "fid, text\n"
        self.vocab_size = vocab_size

        for line in open(path):
            tmp = [x.strip() for x in line.split(',')]
            # takes a list of strings that are space delimited into tokens
            fid = tmp[0]
            sent = tmp[1]
            for w in sent.split():
                self.word_count[w] += 1

        # Easily changed but vocab size is essentially defining your max index
        # 0 is reserved and UNK is reserved so you will get vocab_size-2 to get
        # indices from 0-(vocab_size-1) i.e. vocab_size = 50, we get indices 0-49
        for count, w in enumerate(self.word_count.most_common(self.vocab_size - 2)):
            self.w2i[w[0]] = count + 1
            self.i2w[count + 1] = w[0]
        self.oov_index = min([self.vocab_size - 1, len(self.word_count) + 1])
        self.vocab_size = self.oov_index + 1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    # author: zyf
    # create vocab by \t
    # input: train/valid/test files
    def train_from_file2(self, paths, vocab_size):
        if len(self.word_count) != 0:
            raise Exception("To update existing tokenizer with new vocabulary, run update() or update_from_file()")

        # This is for our file representation of "fid, text\n"
        self.vocab_size = vocab_size

        for path in tqdm(paths):
            for line in open(path):
                tmp = [x.strip() for x in line.split('\t')]
                # takes a list of strings that are space delimited into tokens
                fid = tmp[0]
                sent = tmp[1]
                for w in sent.split():
                    self.word_count[w] += 1

        # Easily changed but vocab size is essentially defining your max index
        # 0 is reserved and UNK is reserved so you will get vocab_size-2 to get
        # indices from 0-(vocab_size-1) i.e. vocab_size = 50, we get indices 0-49

        # atten! we need a <s> and </s> label to express start and end of a sentence
        # fot -4, we used to add <s>, </s>, <UNK> and <NULL>
        for count, w in enumerate(self.word_count.most_common(self.vocab_size - 4)):
            # start with index: 3
            self.w2i[w[0]] = count + 3
            self.i2w[count + 3] = w[0]
        self.oov_index = min([self.vocab_size - 1, len(self.word_count) + 1])
        self.vocab_size = self.oov_index + 1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0

        self.w2i['<s>'] = 1
        self.w2i['</s>'] = 2

        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'
        self.i2w[1] = '<s>'
        self.i2w[2] = '</s>'

    def update(self, texts):
        # takes a list of strings that are space delimited into tokens
        for sent in texts:
            for w in sent.split():
                self.word_count[w] += 1

        # reset w2i and i2w for new vocab
        self.w2i = {}
        self.i2w = {}

        # Easily changed but vocab size is essentially defining your max index
        # 0 is reserved and UNK is reserved so you will get vocab_size-2 to get
        # indices from 0-(vocab_size-1) i.e. vocab_size = 50, we get indices 0-49
        for count, w in enumerate(self.word_count.most_common(self.vocab_size - 2)):
            self.w2i[w[0]] = count + 1
            self.i2w[count + 1] = w[0]
        self.oov_index = min([self.vocab_size - 1, len(self.word_count) + 1])
        self.vocab_size = self.oov_index + 1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    def update_from_file(self, path):
        # takes a list of strings that are space delimited into tokens
        for line in open(path):
            tmp = [x.strip() for x in line.split(',')]
            # takes a list of strings that are space delimited into tokens
            fid = tmp[0]
            sent = tmp[1]
            for w in sent.split():
                self.word_count[w] += 1

        # reset w2i and i2w for new vocab
        self.w2i = {}
        self.i2w = {}

        # Easily changed but vocab size is essentially defining your max index
        # 0 is reserved and UNK is reserved so you will get vocab_size-2 to get
        # indices from 0-(vocab_size-1) i.e. vocab_size = 50, we get indices 0-49
        for count, w in enumerate(self.word_count.most_common(self.vocab_size - 2)):
            self.w2i[w[0]] = count + 1
            self.i2w[count + 1] = w[0]
        self.oov_index = min([self.vocab_size - 1, len(self.word_count) + 1])
        self.vocab_size = self.oov_index + 1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size
        # reset w2i and i2w for new vocab
        self.w2i = {}
        self.i2w = {}

        # Easily changed but vocab size is essentially defining your max index
        # 0 is reserved and UNK is reserved so you will get vocab_size-2 to get
        # indices from 0-(vocab_size-1) i.e. vocab_size = 50, we get indices 0-49
        for count, w in enumerate(self.word_count.most_common(self.vocab_size - 2)):
            self.w2i[w[0]] = count + 1
            self.i2w[count + 1] = w[0]
        self.oov_index = min([self.vocab_size - 1, len(self.word_count) + 1])
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    # maxlen: comment??
    def texts_to_sequences(self, texts, maxlen=None, padding='post', truncating='post', value=0):

        if len(self.word_count) == 0:
            raise Exception("Tokenizer has not been trained, no words in vocabulary.")

        # takes a list of strings that are space delimited into tokens
        all_seq = list()
        for sent in texts:
            seq = []
            for w in sent.split():
                try:
                    seq.append(self.w2i[w])
                except:
                    seq.append(self.oov_index)

                if maxlen is not None:
                    if len(seq) == maxlen:
                        break

            all_seq.append(seq)

        return pad_sequences(all_seq, maxlen=maxlen, padding=padding, truncating=truncating, value=value)

    # authoer: zyf
    # input texts from file into a big list, then convert them into a metrix for training
    def texts_to_sequences_zyf(self, datapath, type, maxlen=None, padding='post', truncating='post', value=0):

        if len(self.word_count) == 0:
            raise Exception("Tokenizer has not been trained, no words in vocabulary.")

        file_type = ['test.code','valid.code','train.code']
        # test/valid/train
        for i in range(len(file_type)):
            data_type = file_type[i].split(".")[0]
            print(data_type)
            with open(datapath + data_type + "/" + file_type[i] + "." + type, 'r') as f:
                data = f.readlines()
            print("dealing with %s" %(file_type[i]+'.'+type))
            file_seqs= list()
            fids = []

            for line in tqdm(data):
                fid, content = line.split("\t",1)
                # every sentence use a list
                seq = []
                # if the data is comment
                # add start label first
                if type=='nl':
                    seq.append(self.w2i['<s>'])
                fids.append(int(fid))
                for w in content.split():
                    try:
                        seq.append(self.w2i[w])
                    except:
                        seq.append(self.oov_index)
                    if maxlen is not None:
                        if len(seq) == maxlen:
                            break
                # if the data is comment
                # add end label last
                if type == 'nl':
                    seq.append(self.w2i['</s>'])
                file_seqs.append(seq)
            result = pad_sequences(file_seqs, maxlen=maxlen, padding=padding, truncating=truncating, value=value)


            final_dict = {}
            print("write ",type + "." + data_type," file...")
            with open(datapath + type + "." + data_type, 'wb') as f:
                for fun_id, seq in zip(fids, result):
                    final_dict[fun_id] = np.array(seq)

                pickle.dump(final_dict, f)

    # make groundtruth data
    # with <s> and </s>
    def add_label_to_comment(self, datapath, type='test'):
        with open(datapath + "nl." + type, 'rb') as fp1:
            # get dict
            index_data = pickle.load(fp1)

        with open(datapath + 'nl-withlabel.' + type, 'w') as fp2:
            for fun_id, comment in index_data.items():
                comment_word = []
                for index in comment.tolist():
                    if index == 0:
                        break
                    else:
                        comment_word.append(self.i2w[index])
                current_comment = " ".join(comment_word)
            fp2.write(str(fun_id) + ', ' + comment_word + '\n')

        print("groundtruth done...")




    def texts_to_sequences_from_file(self, path, maxlen=50, padding='post', truncating='post', value=0):

        if len(self.word_count) == 0:
            raise Exception("Tokenizer has not been trained, no words in vocabulary.")

        all_seq = {}
        for line in open(path):
            tmp = [x.strip() for x in line.split(',')]
            # takes a list of strings that are space delimited into tokens
            fid = int(tmp[0])
            sent = tmp[1]
            # takes a list of strings that are space delimited into tokens
            seq = []
            for w in sent.split():
                try:
                    seq.append(self.w2i[w])
                except:
                    seq.append(self.oov_index)

                if maxlen is not None:
                    if len(seq) == maxlen:
                        break

            all_seq[fid] = seq
        return {key: newval for key, newval in zip(all_seq.keys(),
                                                   pad_sequences(all_seq.values(), maxlen=maxlen, padding=padding,
                                                                 truncating=truncating, value=value))}

    def seq_to_text(self, seq):
        return [self.i2w[x] for x in seq]

    def forw2v(self, seq):
        return [self.i2w[x] for x in seq if self.i2w[x] not in ['<NULL>', '<s>', '</s>']]

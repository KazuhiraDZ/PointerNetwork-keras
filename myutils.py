import sys
from timeit import default_timer as timer
import keras
import numpy as np
import tensorflow as tf
import random
from keras.backend.tensorflow_backend import set_session

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

# TODO refactor this so it imports in the necessary functions
dataprep = '/scratch/funcom/data/standard'
sys.path.append(dataprep)
import tokenizer

start = 0
end = 0

def init_tf(gpu, horovod=False):

    #config = tf.ConfigProto()
    config = tf.ConfigProto(log_device_placement=False ,allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu)
    print("current GPU:{}".format(gpu))
    set_session(tf.Session(config=config))

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer) # i2w
    for i in seq:
        sent.append(check[int(i)])

    return(' '.join(sent))

def seq2sent_withpointer(seq, tokenizer, config, oov_vocabs):
    sent = []
    check = index2word(tokenizer) # i2w

    for i in seq:
        if i >= len(check) and config['use_pointer']:
            cur_token = oov_vocabs[int(i) - len(check)]
            #if cur_token:
            sent.append(cur_token)
        else:
            sent.append(check[int(i)])

    try:
        ans = ' '.join(sent)
    except TypeError as e:
        print(oov_vocabs)
        print(sent)
        print(seq)

    return ans

def source2ids_zyf(source_words, vocab_n):
    """
    a variety for function source2ids
    we take source vocab itself to extend vocab for summary in batch
    """

    ids = []
    current_words_and_oovs = []
    unk_id = vocab_n['<UNK>']
    for w in source_words:
        # used vocab_nl to search
        i = vocab_n[w] if w in vocab_n else unk_id
        if i == unk_id:
            #print(i)
            if w not in current_words_and_oovs:
                current_words_and_oovs.append(w)
            word_num = current_words_and_oovs.index(w)
            ids.append(len(vocab_n) + word_num)
        else: # as source code is another language to summary, we also need to view the input tokens as candidate for summary to copy
            ids.append(i)
    #if current_words_and_oovs:
    #    print(current_words_and_oovs)
    return ids, current_words_and_oovs

class batch_gen(keras.utils.Sequence):
    def __init__(self, seqdata, tt, mt, config):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']
        self.seqdata = seqdata
        self.mt = mt
        self.allfids = list(seqdata['d%s' % (tt)].keys())
        self.num_inputs = config['num_input']
        self.config = config
        
        random.shuffle(self.allfids) # actually, might need to sort allfids to ensure same order

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfids = self.allfids[start:end]

        if self.num_inputs == 2:
            if self.config['use_pointer']:
                return self.divideseqs_without_categorical(batchfids, self.seqdata, self.comvocabsize, self.tt)
            else:
                return self.divideseqs(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.num_inputs == 3:
            if self.config['use_pointer']:
                return self.divideseqs_ast_without_categorical(batchfids, self.seqdata, self.comvocabsize, self.tt)
            else:
                return self.divideseqs_ast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        else:
            return None

    def __len__(self):
        return int(np.ceil(len(list(self.seqdata['d%s' % (self.tt)]))/self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allfids)

    def divideseqs(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        comseqs = list()
        comouts = list()

        for fid in batchfids:
            input_datseq = seqdata['d%s' % (tt)][fid]
            input_comseq = seqdata['c%s' % (tt)][fid]

        limit = -1
        c = 0
        for fid in batchfids:
            wdatseq = seqdata['d%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            
            wdatseq = wdatseq[:self.config['tdatlen']]
            
            for i in range(len(wcomseq)):
                datseqs.append(wdatseq)
                comseq = wcomseq[:i]
                comout = keras.utils.to_categorical(wcomseq[i], num_classes=comvocabsize)
                #comout = np.asarray([wcomseq[i]])
                
                for j in range(0, len(wcomseq)):
                    try:
                        comseq[j]
                    except IndexError as ex:
                        comseq = np.append(comseq, 0)

                comseqs.append(np.asarray(comseq))
                comouts.append(np.asarray(comout))

            c += 1
            if(c == limit):
                break

        datseqs = np.asarray(datseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        return [[datseqs, comseqs], comouts]

    # author: zyf
    def divideseqs_without_categorical(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        datseqs = list()
        comseqs = list()
        #smlseqs = list()
        comouts = list()

        ex_datseqs = list()

        limit = -1
        c = 0

        max_oov = 0
        for fid in batchfids:

            wdatseq = seqdata['d%s' % (tt)][fid]
            wcomseq = seqdata['ex_c%s' % (tt)][fid]  # for extend target
            #wsmlseq = seqdata['s%s' % (tt)][fid]
            woovseq = seqdata['o%s' % (tt)][fid] # maybe [0],[1],[2],[3]
            ex_datseq = seqdata['ex_d%s' % (tt)][fid]

            wdatseq = wdatseq[:self.config['tdatlen']]

            max_oov = max(max_oov, woovseq[0])
            for i in range(0, len(wcomseq)):
                datseqs.append(wdatseq)
                #smlseqs.append(wsmlseq)
                # slice up whole comseq into seen sequence and current sequence
                # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                comseq = wcomseq[0:i]
                comout = wcomseq[i]

                ex_datseqs.append(ex_datseq)
                #comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                # extend length of comseq to expected sequence size
                # the model will be expecting all input vectors to have the same size
                for j in range(0, len(wcomseq)):
                    try:
                        comseq[j]
                    except IndexError as ex:
                        comseq = np.append(comseq, 0)

                comseqs.append(comseq)
                comouts.append(np.asarray(comout))

            c += 1
            if (c == limit):
                break

        oov_seqs = np.asarray(np.ones(shape=(len(batchfids) * len(wcomseq), 1)) * max_oov)

        datseqs = np.asarray(datseqs)
        #smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        ex_datseqs = np.asarray(ex_datseqs)
        return [[datseqs, comseqs, oov_seqs, ex_datseqs], comouts]

    def divideseqs_ast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        comseqs = list()
        smlseqs = list()
        comouts = list()

        limit = -1
        c = 0
        for fid in batchfids:

            wdatseq = seqdata['d%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wdatseq = wdatseq[:self.config['tdatlen']]

            for i in range(0, len(wcomseq)):
                datseqs.append(wdatseq)
                smlseqs.append(wsmlseq)
                # slice up whole comseq into seen sequence and current sequence
                # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                comseq = wcomseq[0:i]
                comout = wcomseq[i]
                comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                # extend length of comseq to expected sequence size
                # the model will be expecting all input vectors to have the same size
                for j in range(0, len(wcomseq)):
                    try:
                        comseq[j]
                    except IndexError as ex:
                        comseq = np.append(comseq, 0)

                comseqs.append(comseq)
                comouts.append(np.asarray(comout))

            c += 1
            if(c == limit):
                break

        datseqs = np.asarray(datseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        return [[datseqs, comseqs, smlseqs], comouts]

    # author: zyf
    def divideseqs_ast_without_categorical(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        datseqs = list()
        comseqs = list()
        smlseqs = list()
        comouts = list()

        ex_datseqs = list()

        limit = -1
        c = 0

        max_oov = 0
        for fid in batchfids:

            wdatseq = seqdata['d%s' % (tt)][fid]
            wcomseq = seqdata['ex_c%s' % (tt)][fid]  # for extend target
            wsmlseq = seqdata['s%s' % (tt)][fid]
            woovseq = seqdata['o%s' % (tt)][fid] # maybe [0],[1],[2],[3]
            ex_datseq = seqdata['ex_d%s' % (tt)][fid]

            wdatseq = wdatseq[:self.config['tdatlen']]

            max_oov = max(max_oov, woovseq[0])
            for i in range(0, len(wcomseq)):
                datseqs.append(wdatseq)
                smlseqs.append(wsmlseq)
                # slice up whole comseq into seen sequence and current sequence
                # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                comseq = wcomseq[0:i]
                comout = wcomseq[i]

                ex_datseqs.append(ex_datseq)
                #comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                # extend length of comseq to expected sequence size
                # the model will be expecting all input vectors to have the same size
                for j in range(0, len(wcomseq)):
                    try:
                        comseq[j]
                    except IndexError as ex:
                        comseq = np.append(comseq, 0)

                comseqs.append(comseq)
                comouts.append(np.asarray(comout))

            c += 1
            if (c == limit):
                break

        oov_seqs = np.asarray(np.ones(shape=(len(batchfids) * len(wcomseq), 1)) * max_oov)

        datseqs = np.asarray(datseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        ex_datseqs = np.asarray(ex_datseqs)
        return [[datseqs, comseqs, smlseqs, oov_seqs, ex_datseqs], comouts]
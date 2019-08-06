import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time

import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

import keras
import keras.utils
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
import keras.backend as K
from model import create_model
from myutils import prep, drop, batch_gen, init_tf, seq2sent

class PlzMyGod(Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epo, logs=None):
        self.model.save('/home/x/mydisk/zyf_Project/funcom4/funcom/data/wx_ICSE/output/models/model_epoch_%d.h5' % epo)

class HistoryCallback(Callback):

    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig

        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                fn = outdir + '/histories/' + self.modeltype + '_hist_' + str(self.timestart) + '.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)

                fn = outdir + '/histories/' + self.modeltype + '_conf_' + str(self.timestart) + '.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='2')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='ast-attendgru')
    parser.add_argument('--with-multigpu', dest='multigpu', action='store_true', default=False)
    parser.add_argument('--data', dest='dataprep', type=str,
                        default='/home/x/mydisk/zyf_Project/funcom-pointerOnly-zyf/funcom/data/zyf_ICSE')
    parser.add_argument('--outdir', dest='outdir', type=str,
                        default='/home/x/mydisk/zyf_Project/funcom-pointerOnly-zyf/funcom/data/zyf_ICSE/output')
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    args = parser.parse_args()

    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    multigpu = args.multigpu
    K.set_floatx(args.dtype)
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel

    sys.path.append(dataprep)
    #print(K.tensorflow_backend._get_available_gpus())

    init_tf(gpu)

    # To Commenting
    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/dats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    drop()

    steps = int(len(seqdata['ctrain']) / batch_size) + 1
    valsteps = int(len(seqdata['cval']) / 100) + 1

    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps * batch_size))
    # print('vaidation data size {}'.format(valsteps * 100))
    # print('------------------------------------------')

    config = dict()

    config['tdatvocabsize'] = tdatvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    config['tdatlen'] = len(list(seqdata['dtrain'].values())[0])
    config['comlen'] = len(list(seqdata['ctrain'].values())[0])
    config['smllen'] = len(list(seqdata['strain'].values())[0])

    config['multigpu'] = multigpu
    config['batch_size'] = batch_size

    # end commenting..

    # continue to train if have a modelfile and configfile
    modelfile = ""

    configfile = outdir + '/histories/' + modeltype + '_conf_' + str(1554375680) + '.pkl'

    if os.path.exists(modelfile) and os.path.exists(configfile):
        prep("loading model...")
        model = keras.models.load_model(modelfile, custom_objects={})
        config = pickle.load(open(configfile, 'rb'))
    else:
        prep('creating model... ')
        config, model = create_model(modeltype, config)
        #config, model = create_model(modeltype)

    drop()

    print(model.summary())

    gen = batch_gen(seqdata, 'train', modeltype, config)
    os.makedirs(outdir + '/models/', exist_ok=True)
    checkpoint = ModelCheckpoint(outdir + '/models/' + modeltype + '_E{epoch:02d}_' + str(timestart) + '.h5', save_weights_only=True)

    #plz = PlzMyGod(model)

    savehist = HistoryCallback()
    savehist.setCatchExit(outdir, modeltype, timestart, config)

    #valgen = batch_gen(seqdata, 'val', modeltype, config)

    callbacks = [checkpoint, savehist]

    try:
        # history = model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=3, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
        history = model.fit_generator(gen, steps_per_epoch=steps, epochs=10, verbose=1, max_queue_size=3,
                                      callbacks=callbacks, validation_data=None, validation_steps=None)
    except Exception as ex:
        pass
        #print(ex)
        #traceback.print_exc(file=sys.stdout)

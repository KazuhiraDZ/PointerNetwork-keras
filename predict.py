import os
import sys
import traceback
import pickle
import argparse
import collections
from keras import metrics
import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from multiprocessing import Pool

from timeit import default_timer as timer

from model import create_model
from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word, init_tf, seq2sent_withpointer
import keras
import keras.backend as K

def gendescr_2inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...
    
    tdats, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)

    for i in range(1, comlen):
        results = model.predict([tdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...
    
    tdats, coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

# [dat, comstart, sml, ex_dat, oov_num])
def gendescr_3inp_withpointer(model, data, comstok, comlen, max_oov, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, coms, smls, ex_dats, oov_nums, oov_tokens = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)
    ex_dats = np.array(ex_dats) # [batch_size, enc_step, ]
    oov_nums = np.ones(shape=(batchsize, 1)) * max_oov

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smls, oov_nums, ex_dats], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com, oov_token in zip(data.keys(), coms, oov_tokens):
        final_data[fid] = seq2sent_withpointer(com, comstok, config, oov_token)

    return final_data



def gendescr_4inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms, smls = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)

    #print(sdats)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('modelfile', type=str, default="/home/x/mydisk/zyf_Project/funcom/funcom/data/baseline-filter/output/models/ast-attendgru_E10_1553822309.h5")
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='2')
    parser.add_argument('--data', dest='dataprep', type=str,
                        default='/home/x/mydisk/zyf_Project/funcom4/funcom/data/wx_ICSE')
    parser.add_argument('--outdir', dest='outdir', type=str,
                        default='/home/x/mydisk/zyf_Project/funcom4/funcom/data/wx_ICSE/output')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=128)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--zero-dats', dest='zerodats', action='store_true', default=False)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')

    args = parser.parse_args()

    outdir = args.outdir
    dataprep = args.dataprep
    use_pointer = True
    modelfile = "/home/x/mydisk/zyf_Project/funcom4/funcom/data/wx_ICSE/output/models/ast-attendgru_E10_1564891031.h5"
    modelfile2 = modelfile.split('/')[-1]


    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    outfile = args.outfile
    zerodats = args.zerodats

    if outfile is None:
        outfile = modelfile.split('/')[-1]

    K.set_floatx(args.dtype)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel

    sys.path.append(dataprep)
    import tokenizer

    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/dats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    drop()

    if zerodats:
        v = np.zeros(100)
        for key, val in seqdata['dtrain'].items():
            seqdata['dttrain'][key] = v

        for key, val in seqdata['dval'].items():
            seqdata['dtval'][key] = v
    
        for key, val in seqdata['dtest'].items():
            seqdata['dtest'][key] = v

    allfids = list(seqdata['cval'].keys())
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    #and_file = "/home/x/mydisk/zyf_Project/funcom/funcom/data/baseline/and_data_index.txt"
    and_file = ""
    if os.path.exists(and_file):
        print("read exist and_file...")
        and_indexs = pickle.load(open(and_file,'rb'))
        datlen = len(seqdata['dval'][list(seqdata['dval'].keys())[0]])
        comlen = len(seqdata['cval'][list(seqdata['cval'].keys())[0]])
        allfids = and_indexs
        smllen = len(seqdata['sval'][list(seqdata['sval'].keys())[0]])
    else:
        print("get data from dataset...")
        datlen = len(seqdata['dval'][list(seqdata['dval'].keys())[0]])
        comlen = len(seqdata['cval'][list(seqdata['cval'].keys())[0]])
        smllen = len(seqdata['sval'][list(seqdata['sval'].keys())[0]])


    prep('loading config... ')
    (modeltype, mid, timestart) = modelfile2.split('_')
    (timestart, ext) = timestart.split('.')
    modeltype = modeltype.split('/')[-1]



    config = dict()

    config['tdatvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    config['tdatlen'] = len(list(seqdata['dval'].values())[0])
    config['comlen'] = len(list(seqdata['cval'].values())[0])
    config['smllen'] = len(list(seqdata['sval'].values())[0])

    config['multigpu'] = False
    config['batch_size'] = batchsize

    #config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+timestart+'.pkl', 'rb'))
    #config = seqdata["config"]

    num_inputs = 3
    drop()

    config, model = create_model(modeltype, config)
    prep('loading model... ')
    #model = keras.models.load_model(modelfile, custom_objects={})
    model.load_weights(modelfile)
    print(model.summary())
    drop()

    comstart = np.zeros(comlen)
    st = comstok.w2i['<s>']
    comstart[0] = st
    outfn = outdir+"/predictions/predict-val-{}.txt".format(outfile.split('.')[0])
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    print("all files is: ",len(allfids))
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
 
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        batch = {}
        st = timer()
        max_oov = 0
        for fid in fid_set:
            dat = seqdata['dval'][fid]
            sml = seqdata['sval'][fid]
            ex_dat = seqdata['ex_dval'][fid]
            oov_num = seqdata['oval'][fid]
            oov_tokens = seqdata['ovval'][fid]
            max_oov = max(max_oov, oov_num)
            # adjust to model's expected data size
            dat = dat[:50]
            sml = sml[:10]
            ex_dat = ex_dat[:50]

            if num_inputs == 2:
                batch[fid] = np.asarray([dat, comstart])
            elif num_inputs == 3 and not use_pointer:
                batch[fid] = np.asarray([dat, comstart, sml])
            elif num_inputs == 3 and use_pointer:
                batch[fid] = np.asarray([dat, comstart, sml, ex_dat, oov_num, oov_tokens])
            else:
                print('error: invalid number of inputs specified')
                sys.exit()

        if num_inputs == 2:
            batch_results = gendescr_2inp(model, batch, comstok, comlen, batchsize, config, strat='greedy')
        elif num_inputs == 3 and not use_pointer:
            batch_results = gendescr_3inp(model, batch, comstok, comlen, batchsize, config, strat='greedy')
        elif num_inputs == 3 and use_pointer:
            batch_results = gendescr_3inp_withpointer(model, batch, comstok, comlen, max_oov, batchsize, config, strat='greedy')
        else:
            print('error: invalid number of inputs specified')
            sys.exit()

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()

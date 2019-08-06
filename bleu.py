import sys
import pickle
import argparse
import re
import os
from nltk.translate.bleu_score import SmoothingFunction

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from funcom.rouge.rouge import Rouge
from funcom.meteor.meteor import Meteor
from funcom.cider.cider import Cider
from myutils import prep, drop


def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret


def bleu_so_far(preds, refs, smooth=""):
    # pred: list[['a','b']
    chencherry = SmoothingFunction()
    if smooth != "":
        Ba = corpus_bleu(refs, preds, smoothing_function=chencherry.method1)
    else:
        Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    B2 = corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    B3 = corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    B4 = corpus_bleu(refs, preds, weights=(0, 0, 0, 1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))

    # print(ret)

    otherEva(refs, preds)
    return ret


def otherEva(references, preds):
    gts = {}
    cands = {}

    for i in range(len(preds)):
        # print(references[i])
        cands[i] = [" ".join(preds[i])]
        gts[i] = [" ".join(references[i][0])]

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, cands)
    score_Rouge, scores_Rouge = Rouge().compute_score(gts, cands)
    score_Cider, scores_Cider = Cider().compute_score(gts, cands)

    print("Meteor: ", score_Meteor)
    print("ROUGe: ", score_Rouge)
    print("Cider: ", score_Cider)

    #ba = bleu_so_far(preds, references)
    #return ba

def count_metric_by_token_length(file_path, type="token"):
    files = os.listdir(os.path.join(file_path,type))
    for file in files:
        print("we evaluation file: {}".format(file))
        _, cur_num = file.split("_")
        temp_refs = []
        temp_preds = []
        with open(os.path.join(file_path,type,file), "r") as f:
            data = f.readlines()
        for item in data:
            item = eval(item)
            temp_funid = item["fun_id"]
            temp_pred = item["prediction"]
            temp_ref = item["reference"]
            temp_refs.append(temp_ref)
            temp_preds.append(temp_pred)
        print("computing {} of {} ...".format(type, cur_num))
        if type=="token":
            smooth = ""
        elif type=="nl":
            smooth = "method1"
        print(otherEva_fjk(temp_preds, temp_refs, smooth=smooth))

def otherEva_fjk(preds, references, smooth=""):
    gts = {}
    cands = {}

    for i in range(len(preds)):
        cands[i] = [preds[i]]
        gts[i] = [references[i]]

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, cands)
    score_Rouge, scores_Rouge = Rouge().compute_score(gts, cands)
    score_Cider, scores_Cider = Cider().compute_score(gts, cands)

    print("Meteor: ", score_Meteor)
    print("ROUGe: ", score_Rouge)
    print("Cider: ", score_Cider)

    ba = bleu_so_far_fjk(preds, references, smooth)
    return ba, 0.0

def bleu_so_far_fjk(preds, refs, smooth=""):
    # pred: list[['a','b']
    new_refs = []
    new_preds = []
    for pred, ref in zip(preds, refs):
        temp_pred = pred.split()
        temp_pred = fil(temp_pred)
        temp_ref = ref.split()
        temp_ref = fil(temp_ref)
        new_preds.append(temp_pred)
        new_refs.append([temp_ref])

    chencherry = SmoothingFunction()
    if smooth != "":
        Ba = corpus_bleu(new_refs, new_preds, smoothing_function=chencherry.method1)
    else:
        Ba = corpus_bleu(new_refs, new_preds)
    B1 = corpus_bleu(new_refs, new_preds, weights=(1, 0, 0, 0))
    B2 = corpus_bleu(new_refs, new_preds, weights=(0, 1, 0, 0))
    B3 = corpus_bleu(new_refs, new_preds, weights=(0, 0, 1, 0))
    B4 = corpus_bleu(new_refs, new_preds, weights=(0, 0, 0, 1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('for %s functions\n' % (len(new_preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))

    print(ret)

    #otherEva(refs, preds)
    return Ba

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('input', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str,
                        default='/home/x/mydisk/zyf_Project/funcom4/funcom/data/wx_ICSE')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/scratch/funcom/data/outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = "/home/x/mydisk/zyf_Project/funcom4/funcom/data/wx_ICSE/output/predictions/predict-val-ast-attendgru_E10_1564891031.txt"
    challenge = args.challenge
    obfuscate = args.obfuscate
    sbt = args.sbt

    compare_input_path = "/home/x/mydisk/zyf_Project/funcom4/funcom/data/RQ1-SBT-AO-OUR/test/test_split/"

    if challenge:
        dataprep = '../data/challengeset/output'

    if obfuscate:
        dataprep = '../data/obfuscation/output'

    if sbt:
        dataprep = '../data/sbt/output'

    if input_file is None:
        print('Please provide an input file to test with --input')
        exit()

    sys.path.append(dataprep)
    import tokenizer

    # use dict to align the data
    # because it may be different when we use dict
    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
    predicts.close()
    drop()

    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

    # refs = list()
    # newpreds = list()
    # d = 0
    #
    # select_targets = {}
    # targets = open('%s/test.code.nl' % (dataprep), 'r')
    # for line in targets:
    #     (fid, com) = line.split('\t')
    #     fid = int(fid)
    #     com = com.split()
    #     com = fil(com)
    #
    #     try:
    #         newpreds.append(preds[fid])
    #     except KeyError as ex:
    #         continue
    #
    #     refs.append([com])

    mode = 1
    if mode == 1:
        refs = list()
        newpreds = list()
        d = 0

        select_targets = {}
        targets = open('%s/valid.token.nl' % (dataprep), 'r')
        for line in targets:
            (fid, com) = line.split('\t')
            fid = int(fid)
            com = com.split()
            com = fil(com)

            try:
                newpreds.append(preds[fid])
            except KeyError as ex:
                continue

            refs.append([com])


        print('final status')
        print(bleu_so_far(newpreds, refs))

    #count_metric_by_token_length(compare_input_path, "nl")

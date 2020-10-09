#!/usr/bin/env python3

# Predict with multilingual text classifier.

import sys
import os
import json

import numpy as np

from logging import warning, error

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score

from common_multilabel import load_fasttext_data_multi
from common_multilabel import text_to_indices, label_to_index
from common_multilabel import transform2mlb_val
from common import NullOutput
from joblib import dump, load
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve,average_precision_score,auc,precision_recall_fscore_support,classification_report


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-q', '--quiet', default=False, action='store_true',
                    help='suppress messages')
    ap.add_argument('model', metavar='MODEL',
                    help='path to trained model')
    ap.add_argument('data', metavar='LANG:PATH',
                    help='data in language LANG (fastText format)')
    ap.add_argument('binlabels', 
                    help="the label binarizer model, mlb")
    ap.add_argument('-T', '--threshold', default=0.5,
                    help='round to 1 or 0')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    log = sys.stderr if not args.quiet else NullOutput()
    modelfn = '{}.h5'.format(args.model)
    metafn = '{}.json'.format(args.model)
    model = load_model(modelfn)
    print('Loaded model from {}'.format(modelfn), file=log)
    with open(metafn, encoding='utf-8') as f:
        meta = json.load(f)
    try:
        cased = meta['cased']
        max_length = meta['max_length']
#        label_to_idx = meta['label_to_idx']
        token_to_idx = meta['token_to_idx']
    except KeyError:
        raise ValueError('incomplete metadata in {}'.format(metafn))
    print('Loaded metadata from {}'.format(metafn), file=log)

    lang, path = args.data.split(':')
    texts, labels = load_fasttext_data_multi([lang], [path], not cased, log)
    text_indices = [text_to_indices(t, token_to_idx, False) for t in texts]


    print("indexing validation", file=sys.stderr, flush=True)

    binmodel = load(args.binlabels)

    test_label_indices= transform2mlb_val(labels,binmodel)
    test_Y =  test_label_indices

    print("Loaded  {} texts".format(len(text_indices)))

    test_X = pad_sequences(text_indices, maxlen=max_length,
                           padding='post', truncating='post')

    num_classes=len(binmodel.classes_)

    preds = model.predict(test_X, verbose=0)
    th = float(args.threshold)

    preds[preds>=th] = 1
    preds[preds<th]=0

    probabs = model.predict(test_X)
    prec = dict()
    rec = dict()
    average_prec = dict()

    print(classification_report(test_Y,preds))

    for i in range(num_classes):
        prec[i], rec[i], _ = precision_recall_curve(test_Y[:,i], probabs[:,i])
        average_prec[i] = average_precision_score(test_Y[:,i],probabs[:,i])

    prec["micro"],rec["micro"], _ = precision_recall_curve(test_Y.ravel(),probabs.ravel())

    average_prec["micro"] = average_precision_score(test_Y,probabs,average="micro")
 
    print("AUC IS", metrics.auc(rec["micro"], prec["micro"]))

    return 0


if __name__ == '__main__':
    try:
        retval = main(sys.argv)
    except ValueError as e:
        error(e)
        retval = -1
    sys.exit(retval)

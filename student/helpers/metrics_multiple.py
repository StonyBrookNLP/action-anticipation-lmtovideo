import sys
import os
import os.path as osp
import glob
from collections import OrderedDict
from collections.abc import Iterable
import json
import subprocess
import pickle
import logging
import h5py
import math
import operator
import pathlib

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import numpy as np


class EmptyResdirError(ValueError):
    pass


def allkeys(obj, keys=[]):
    """Recursively find all leaf keys in h5. """
    keys = []
    for key in obj.keys():
        if isinstance(obj[key], h5py.Group):
            keys += [f'{key}/{el}' for el in allkeys(obj[key])]
        else:
            keys.append(key)
    return keys


def gen_load_resfiles(resdir):
    resfiles = glob.glob(osp.join(resdir, '*.pth'))
    if len(resfiles) == 0:
        resfiles = glob.glob(osp.join(resdir, '*.h5'))
    if len(resfiles) == 0:
        raise EmptyResdirError(f'Didnt find any resfiles in {resdir}')
    for resfile in resfiles:
        if resfile.endswith('.pth'):
            output_dict = {
                key: val.numpy() if torch.torch.is_tensor(val) else val
                for key, val in torch.load(resfile).items()
            }
        else:
            output_dict = {}
            with h5py.File(resfile, 'r') as fin:
                for key in allkeys(fin):
                    try:
                        output_dict[key] = fin[key][()]
                    except AttributeError as err:
                        # Happens for the string keys... need to figure what
                        # to do here
                        logging.warning('Unable to load %s (%s)', key, err)
        yield output_dict


def read_results(resdir):
    data = next(gen_load_resfiles(resdir))
    # TODO allow to read only certain keys, eg some times we only need logits
    # which would be faster to read
    res_per_layer = {
        key: OrderedDict()
        for key in data if key not in ['epoch']
    }
    if len(res_per_layer) == 0:
        raise ValueError('No logits found in the output. Note that code was '
                         'changed Aug 26 2020 that renames "output" to '
                         '"logits" etc. So might need to rerun testing.')
    logging.info('Reading from resfiles')
    for data in gen_load_resfiles(resdir):
        for i, idx in enumerate(data['idx']):
            idx = int(idx)
            for key in res_per_layer:
                if idx not in res_per_layer[key]:
                    res_per_layer[key][idx] = []
                res_per_layer[key][idx].append(data[key][i])
    # Mean over all the multiple predictions per key
    final_res = {}
    for key in res_per_layer:
        if len(res_per_layer[key]) == 0:
            continue
        max_idx = max(res_per_layer[key].keys())
        key_output = np.zeros([
            max_idx + 1,
        ] + list(res_per_layer[key][0][0].shape))
        for idx in res_per_layer[key]:
            key_output[idx] = np.mean(np.stack(res_per_layer[key][idx]),
                                      axis=0)
        final_res[key] = key_output
    return final_res


def get_pred_for_uid(data, uid):
    data_by_uid = {}
    ind = np.where(data['uid'] == uid)
    data_by_uid['logits'] = data['logits/action'][ind]
    data_by_uid['target'] = data['target/action'][ind]
    data_by_uid['pred_classes'] = np.flip(np.argsort(data['logits/action'][ind]))
    return data_by_uid


def extract_preds(data):
    preds = {
        'uid': data['uid'],
        'logits': data['logits/action'],
        'target': data['target/action']
    }
    return preds


if __name__ == "__main__":
    # filename = "./0.h5"
    # with h5py.File(filename, "r") as f:
    #     # List all groups
    #     keys = f.keys()
    #     print("Keys: %s" % keys)
    #     data = [list(f[k]) for k in keys]
    #
    #     print(len(data))
    # results_dirs = ['/Users/tanviaggarwal/Desktop/SOP/model_ek55/ens_kldiv_top50_6may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/notpretrained_ens_mean_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_ens_attention_nh4_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_ens_mean_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/notpretrained_alberta_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/notpretrained_roberta_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/notpretrained_electra_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/notpretrained_bert_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/notpretrained_distillbert_kldiv_top50_13may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_alberta_kldiv_top50_9may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_roberta_kldiv_top50_9may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_electra_kldiv_top50_9may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_bert_kldiv_top50_9may_wt20_t5/results',
    #                 '/Users/tanviaggarwal/Desktop/SOP/model_ek55/pretrained_distillbert_kldiv_top50_9may_wt20_t5/results'
    #                 ]
    # preds_paths = ['./ek55_preds/notpretrained_ens_attention_nh4_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/notpretrained_ens_mean_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_ens_attention_nh4_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_ens_mean_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/notpretrained_alberta_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/notpretrained_roberta_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/notpretrained_electra_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/notpretrained_bert_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/notpretrained_distillbert_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_alberta_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_roberta_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_electra_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_bert_kldiv_top50_wt20_t5_preds.pickle',
    #                './ek55_preds/pretrained_distillbert_kldiv_top50_wt20_t5_preds.pickle'
    #                ]

    results_dirs = ['/Users/tanviaggarwal/Desktop/SOP/model_egtea/notpretrained_ens_attention_nh4_kldiv_14may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/notpretrained_ens_mean_kldiv_14may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_ens_attention_nh4_kldiv_14may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_ens_mean_kldiv_14may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_alberta_kldiv_11may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_roberta_kldiv_11may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_electra_kldiv_11may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_bert_kldiv_11may_wt150_t10/results',
                    '/Users/tanviaggarwal/Desktop/SOP/model_egtea/pretrained_distillbert_kldiv_11may_wt150_t10/results',
                    ]
    preds_paths = ['./egtea_preds/notpretrained_ens_attention_nh4_kldiv_wt150_t10_preds.pickle',
                   './egtea_preds/notpretrained_ens_mean_kldiv_wt150_t10_preds.pickle',
                   './egtea_preds/pretrained_ens_attention_nh4_kldiv_wt150_t10_preds_preds.pickle',
                   './egtea_preds/pretrained_ens_mean_kldiv_wt150_t10_preds_preds.pickle',
                   './egtea_preds/pretrained_alberta_kldiv_wt150_t10_preds.pickle',
                   './egtea_preds/pretrained_roberta_kldiv_wt150_t10_preds.pickle',
                   './egtea_preds/pretrained_electra_kldiv_wt150_t10_preds.pickle',
                   './egtea_preds/pretrained_bert_kldiv_wt150_t10_preds.pickle',
                   './egtea_preds/pretrained_distillbert_kldiv_wt150_t10_preds.pickle'
                   ]
    for i in range(len(results_dirs)):
        data = read_results(results_dirs[i])
        preds = extract_preds(data)
        with open(preds_paths[i], 'wb') as handle:
            pickle.dump(preds, handle)
    print("processed")
    # data_uid_2 = get_pred_for_uid(data, 19)
    # print(len(data_uid_2))

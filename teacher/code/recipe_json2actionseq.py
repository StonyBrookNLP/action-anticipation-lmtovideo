''' CODE TO CREATE RECIPE ACTION SEQUECE FROM THE INFORMATION-EXTRACTION OUTPUT '''


import json
import os
from copy import deepcopy as cc
import time
import argparse
import json
from typing import List
from collections import Counter
from tqdm import tqdm
import gc, pdb
gc.disable()
from collections import Counter

verb_counter = Counter([])
tuple_len_counter = Counter([])
verb_to_be_removed = ['is', 'are', 'will', 'was', 'be', 'have', 'has', 'let', 'using', 'see', 'using', 'tell', 'may', 'can', 'could' 'want', 'got', "'s", 'been', 'must', 'll' ]


def create_action_seq(action_list):
    '''
    IP: [[[verb_0], [obj_0]], [[verb_1], [obj_1]], [[verb_1], [obj_1]]]
    - Only consider those actions which have 'object'
    - Lemmatize the verb and the nouns
    - Remove stopwords from noun/obj
    - remover actions which have frequently occuring verb (such as is, are, will, be, have, has, ..)
    '''
    global verb_counter, tuple_len_counter, verb_to_be_removed
    
    op = []
    if action_list != []:
        for act in action_list:
            if ( (act[0] != []) and (act[1] != []) and (act[0][0] not in verb_to_be_removed)):
                verb = act[0][0]
                op.append(((verb, act[1][0])))
                tuple_len_counter.update({ len(act) : 1 })
                verb_counter.update({ verb: 1 }) 
    op_res = [" <v> ".join(tup) for tup in op]
    op_res = " <a> ".join(op_res)
    return op_res

data_paths = ['recipe_ie_out_750001_to_1029720.json', 'recipe_ie_out_1_to_750000.json']

recipe_corpus = []
for data_path in data_paths[::-1]:
    print(f'>> File being processed :: {data_path}')    

    data = pd.read_json(os.path.join('./out', data_path))

    processed_ie_op_list = []
    # for idx, row in tqdm(data.iterrows(), total = data.shape[0]):
    for idx, row in data.iterrows():
        if idx % 10000 == 0:
            print(f'>> In iteration no. {idx} out of {data.shape[0]}')
        processed_ie_op = create_action_seq(row.ie_op)
        processed_ie_op_list.append(processed_ie_op)

    data['processed_ie_op'] = pd.Series(processed_ie_op_list)
    data = data.loc[data.processed_ie_op != '']
    data = data.dropna()

    data = data.groupby('id')['processed_ie_op'].agg({' <a> '.join}).reset_index()

    recipe_corpus.extend(list(data['join'].to_list()))

print(f'DONE PROCESSING RECIPE DATA')
op_file_name = open('./out/recipe_action_copus.txt', 'w')
op_file_name.write('\n'.join(recipe_corpus))
op_file_name.close()


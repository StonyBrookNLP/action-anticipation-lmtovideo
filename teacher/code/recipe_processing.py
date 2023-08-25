""" CODE FOR RUNNING THE ALLEN-AI-INFORMATION-EXTRACTION OF THE RECIPE DATASET """

from allennlp.predictors.predictor import Predictor
from typing import List
from tqdm import tqdm
from typing import List
import torch
from tqdm import tqdm
import pandas as pd
import gc
import argparse
import json
gc.disable()

def parse():
    parser = argparse.ArgumentParser(description = "Vert-object extraction from recipe dataset")
    parser.add_argument('-st_idx', type=int, default = 0, help = "Start index of the recipe from which action is to extracted")
    parser.add_argument('-end_idx', type=int, help = "End index of the recipe from which action is to extracted")
    parser.add_argument('-batch_size', type = int, default= 200, help="Batch size")
    args = parser.parse_args()
    return args

# DEFINING THE INFORMATION EXTRACTION MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ie_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
ie_predictor._model = ie_predictor._model.to(device)


def tuple_from_IE_res_v3(tokens: List[str], tags: List[str]) -> str:
    """
    Converts a list of model outputs (i.e., a list of lists of bio tags, each
    pertaining to a single word), returns an inline bracket representation of
    the prediction.
    """
    frame = []
    chunk = []
    verb = []
    obj = []

    for (token, tag) in zip(tokens, tags):
        if (tag == 'I-V') or (tag == 'I-ARG1')  :
            chunk.append(token)
        else:
            if chunk:
                if chunk[0].startswith('V: '):
                    verb.append(" ".join(chunk).replace('V: ', ''))
                else:
                    obj.append(" ".join(chunk).replace('ARG1: ', ''))

                frame.append("[" + " ".join(chunk) + "]")
                chunk = []

            if (tag == 'B-V') or (tag == 'B-ARG1'):
                chunk.append(tag[2:] + ": " + token)
                
    if chunk:
        frame.append("[" + " ".join(chunk) + "]")
    return " ".join(frame), (verb, obj)


if __name__ == '__main__':

    args = parse()

    data_df = pd.read_json('./data/recipe1M_layers/layer1.json')
    data_df = data_df.loc[:, ['title', 'id', 'instructions']]
    print(args)

    if args.end_idx == None:
        end_ix = data_df.shape[0]
    else:
        end_ix = args.end_idx

    print(f'Start index :: {args.st_idx} End index :: {end_ix}')

    data_df = data_df[args.st_idx:end_ix]
    BATCH_LEN = args.batch_size
    ip_buffer = []
    op_final = []
    row_count = 0

    # Iterate over the each recipe
    # for idx_recipe, row in tqdm(data_df.iterrows(), total = (end_ix - args.st_idx) ):
    for idx_recipe, row in data_df.iterrows():

        if row_count % 100 == 0:
            print(f'>> Recipe number :: {row_count}')

        # Create a list of instractions for each recipe
        ie_ip = map(lambda x: {'id': row['id'], 's_no': x[0], 'sentence': x[1]['text'].lower().strip()}, enumerate(row['instructions']))
        ie_ip = list(ie_ip)
        
        ip_buffer.extend(ie_ip)

        # When the size of collected instrction list is == BATCH LEN, pass it through the IE model
        if len(ip_buffer) >= BATCH_LEN:
            op_dict = []

            # pass the input batch of sentneces/instructions through the IE extractor
            op_recipe_batch = ie_predictor.predict_batch_json(ip_buffer)

            # For each sentence in the instruction/sentence batch
            for idx, op_recipe in enumerate(op_recipe_batch):
                tokenized_instr = op_recipe['words']
                verb_data = []

                # For each verb in the list of verb extracted from a sentence
                for verb_info in op_recipe['verbs']:
                    _, verb_obj_tuple = tuple_from_IE_res_v3(tokens = tokenized_instr, tags = verb_info['tags'])
                    verb_data.append(verb_obj_tuple)

                op_dict.append({'id': ip_buffer[idx]['id'], 'sentence': ip_buffer[idx]['sentence'], 's_no': ip_buffer[idx]['s_no'],  'ie_op': verb_data})

            op_final.extend(op_dict)
            ip_buffer = []
            
        row_count += 1
    
    # Clear the reamining input buffer
    op_dict = []
    op_recipe_batch = ie_predictor.predict_batch_json(ip_buffer)

    # For each sentence in the instruction/sentence batch
    for idx, op_recipe in enumerate(op_recipe_batch):
        tokenized_instr = op_recipe['words']
        verb_data = []

        # For each verb in the list of verb extracted from a sentence
        for verb_info in op_recipe['verbs']:
            _, verb_obj_tuple = tuple_from_IE_res_v3(tokens = tokenized_instr, tags = verb_info['tags'])
            verb_data.append(verb_obj_tuple)

        op_dict.append({'id': ip_buffer[idx]['id'], 'sentence': ip_buffer[idx]['sentence'], 's_no': ip_buffer[idx]['s_no'],  'ie_op': verb_data})
    op_final.extend(op_dict)

    print(op_final[0])
    out_file = f'./out/recipe_ie_out_{args.st_idx+1}_to_{end_ix}.json'

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(op_final, f, ensure_ascii=False, indent=1)

import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModel, AutoTokenizer, AdamW, AutoConfig, 
                          BertTokenizerFast, RobertaTokenizerFast, DistilBertTokenizerFast,
                          DebertaTokenizerFast, ElectraTokenizerFast, AlbertTokenizerFast,
                          RobertaModel, BertModel, DistilBertModel, 
                          ElectraModel, DebertaModel, AlbertModel)
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
import pdb
from tqdm import tqdm
import torch.nn as nn
import pickle
import argparse
import os
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight as cls_weigh
from warnings import simplefilter

# POINT TO BE NOTED FOR TOP@5 RECALL FOR MANY SHOT
# The Top-5 recall for a given class c is defined as the fraction of samples of ground truth class c for which the class c is in
# the list of the top-5 anticipated actions [11]. The mean Top-5 Recall is obtained by averaging the Top-5 recall values over
# classes. When evaluating on EPIC-Kitchens, Top-5 Recalls are averaged over the provided list of many-shot verbs, nouns and actions

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# =======================================================
# Defining the teacher model for EGTEA++ dataset
# =======================================================

class egtea_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'roberta':
            print('>> Model type is :: {RoBERTa}')

            if (args.checkpoint_path != None) and (args.checkpoint_path != 'base'):
                print(f'>-> Loading from checkpoint :: {args.checkpoint_path}')
                self.LM_base = RobertaModel.from_pretrained(args.checkpoint_path)
            if args.checkpoint_path == 'base':
                print(f'>-> Loading from checkpoint from standard - checkpoint')
                self.LM_base = RobertaModel.from_pretrained(args.config_path)
            if args.checkpoint_path == None:
                print(f'>-> Random weight initialization')
                self.LM_base = RobertaModel.from_pretrained(args.config_path)
                self.LM_base.init_weights()
        
        if args.model_type == 'bert':
            print('>> Model type is :: {BERT}')
            if (args.checkpoint_path != None) and (args.checkpoint_path != 'base'):
                print(f'>-> Loading from checkpoint :: {args.checkpoint_path}')
                self.LM_base = BertModel.from_pretrained(args.checkpoint_path)
            if args.checkpoint_path == 'base':
                print(f'>-> Loading from checkpoint from standard - checkpoint')
                self.LM_base = BertModel.from_pretrained(args.config_path)
            if args.checkpoint_path == None:
                print(f'>-> Random weight initialization')
                self.LM_base = BertModel.from_pretrained(args.config_path)
                self.LM_base.init_weights()

        if args.model_type == 'distillbert':
            print('>> Model type is :: {Distilbert}')

            if (args.checkpoint_path != None) and (args.checkpoint_path != 'base'):
                print(f'>-> Loading from checkpoint :: {args.checkpoint_path}')
                self.LM_base = DistilBertModel.from_pretrained(args.checkpoint_path)
            if args.checkpoint_path == 'base':
                print(f'>-> Loading from checkpoint from standard - checkpoint')
                self.LM_base = DistilBertModel.from_pretrained(args.config_path)
            if args.checkpoint_path == None:
                print(f'>-> Random weight initialization')
                self.LM_base = DistilBertModel.from_pretrained(args.config_path)
                self.LM_base.init_weights()
        
        if args.model_type == 'electra':
            print('>> Model type is :: {electra}')
            if (args.checkpoint_path != None) and (args.checkpoint_path != 'base'):
                print(f'>-> Loading from checkpoint :: {args.checkpoint_path}')
                self.LM_base = ElectraModel.from_pretrained(args.checkpoint_path)
            if args.checkpoint_path == 'base':
                print(f'>-> Loading from checkpoint from standard - checkpoint')
                self.LM_base = ElectraModel.from_pretrained(args.config_path)
            if args.checkpoint_path == None:
                print(f'>-> Random weight initialization')
                self.LM_base = ElectraModel.from_pretrained(args.config_path)
                self.LM_base.init_weights()

        if args.model_type == 'deberta':
            print('>> Model type is :: {deberta}')
            if (args.checkpoint_path != None) and (args.checkpoint_path != 'base'):
                print(f'>-> Loading from checkpoint :: {args.checkpoint_path}')
                self.LM_base = DebertaModel.from_pretrained(args.checkpoint_path)
            if args.checkpoint_path == 'base':
                print(f'>-> Loading from checkpoint from standard - checkpoint')
                self.LM_base = DebertaModel.from_pretrained(args.config_path)
            if args.checkpoint_path == None:
                print(f'>-> Random weight initialization')
                self.LM_base = DebertaModel.from_pretrained(args.config_path)
                self.LM_base.init_weights()

        if args.model_type == 'alberta':
            print('>> Model type is :: {albert}')
            if (args.checkpoint_path != None) and (args.checkpoint_path != 'base'):
                print(f'>-> Loading from checkpoint :: {args.checkpoint_path}')
                self.LM_base = AlbertModel.from_pretrained(args.checkpoint_path)
            if args.checkpoint_path == 'base':
                print(f'>-> Loading from checkpoint from standard - checkpoint')
                self.LM_base = AlbertModel.from_pretrained(args.config_path)
            if args.checkpoint_path == None:
                print(f'>-> Random weight initialization')
                self.LM_base = AlbertModel.from_pretrained(args.config_path)
                self.LM_base.init_weights()
        
        self.lin_trf = nn.Linear(768, 2048)
        self.out_action = nn.Linear(768, args.no_action_classes)
        self.out_verb = nn.Linear(768, args.no_verb_classes)
        self.out_noun = nn.Linear(768, args.no_noun_classes)

    def forward(self, input_id, attn_mask):
        LM_op = self.LM_base(input_ids = input_id, attention_mask = attn_mask, return_dict = True) 
        LM_op_cls = LM_op['last_hidden_state'][:, 0, :]
        # LM_op_cls= self.lin_trf(LM_op_cls)

        output_action = self.out_action(LM_op_cls)    # dim --> (B, C)
        output_verb = self.out_verb(LM_op_cls)        # dim --> (B, C)
        output_noun = self.out_noun(LM_op_cls)        # dim --> (B, C)

        return {'logit_action': output_action, 'logit_verb': output_verb, 
                'logit_noun': output_noun, 'LM_feat': LM_op_cls}

# =======================================================
# Defining the 50 salad dataset
# =======================================================

class epic55_dataset(Dataset):
    def __init__(self, act_id2txt_mapping, data_path, args):

        # reading the lable sequence file
        with open(data_path, "rb") as input_file:
            self.full_dataset = pickle.load(input_file)
        
        if args.model_type == 'roberta':
            self.tokenizer = RobertaTokenizerFast.from_pretrained(args.config_path)
        if args.model_type == 'bert':
            self.tokenizer = BertTokenizerFast.from_pretrained(args.config_path)
        if args.model_type == 'distillbert':
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(args.config_path)
        if args.model_type == 'electra':
            self.tokenizer = ElectraTokenizerFast.from_pretrained(args.config_path)
        if args.model_type == 'deberta':
            self.tokenizer = DebertaTokenizerFast.from_pretrained(args.config_path)
        if args.model_type == 'alberta':
            self.tokenizer = AlbertTokenizerFast.from_pretrained(args.config_path)

        self.op_label_action, self.op_label_verb, self.op_label_noun = [], [], []
        self.ip_txt_list = []
        self.uid = []
        
        for idx, seg_dict in enumerate(self.full_dataset):
            seg_history_list = seg_dict['history'] 
            target = seg_dict['target']
            for segment_seq in seg_history_list:
                
                # Only use the last K segments of the history sequence
                ip_txt = [act_id2txt_mapping[id]['action_txt'] for id in segment_seq[(-args.hist_len)::1]]
                ip_txt = " <a> ".join(ip_txt)

                self.ip_txt_list.append(ip_txt)
                self.op_label_action.append( target )
                self.op_label_verb.append( act_id2txt_mapping[target]['verb_class'] )
                self.op_label_noun.append( act_id2txt_mapping[target]['noun_class'] )
                self.uid.append( seg_dict['uid'] )

        # Encoding the text
        self.batch_encodings = self.tokenizer.batch_encode_plus(self.ip_txt_list)
        self.batch_encodings.input_ids = [torch.tensor(txt[(-args.max_len)::1]) for txt in self.batch_encodings.input_ids]
        self.batch_encodings.attention_mask = [torch.tensor([1]*len(txt_enc)) for txt_enc in self.batch_encodings.input_ids]

        self.batch_encodings.input_ids = pad_sequence(self.batch_encodings.input_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id)
        self.batch_encodings.attention_mask = pad_sequence(self.batch_encodings.attention_mask, batch_first=True, padding_value=0)

        self.op_label_action = torch.tensor( self.op_label_action )
        self.op_label_verb = torch.tensor( self.op_label_verb )
        self.op_label_noun = torch.tensor( self.op_label_noun )
        self.uid = torch.tensor( self.uid  )

    def __len__(self):
        return len(self.batch_encodings.input_ids)

    def __getitem__(self, idx):
        return {'uid': self.uid[idx], 
                'input_id': self.batch_encodings.input_ids[idx], 
                'attn_mask' : self.batch_encodings.attention_mask[idx], 
                'label_action': self.op_label_action[idx],
                'label_verb': self.op_label_verb[idx],
                'label_noun': self.op_label_noun[idx]}

# =======================================================
# Compute performance metrics from list of logits and ground-truth
# =======================================================

def compute_performance(true, pred_logit, args):
    top_k_score = {}
    pred = [p.argmax() for p in pred_logit]

    # Computing the top 5 accuracy
    for k in range(1, 6):
        topK_acc = sum([g in np.argsort(-p)[:k] for p, g in zip(pred_logit, true)])/len(true)
        top_k_score[k] = topK_acc

    # Computing the recall@5 - MANY SHOT
    # topK_acc_c = {}
    # for c in many_shot_action_class_list:
    #     pred_logit_c = [p for ix, p in enumerate(pred_logit) if true[ix] == c]
    #     true_c = [p for ix, p in enumerate(true) if true[ix] == c]
    #     topK_acc_c[c] = sum([g in np.argsort(-p)[:5] for p, g in zip(pred_logit_c, true_c)])/len(true_c)
    # recall_many_shot = np.array([v for k, v in topK_acc_c.items()])

    # Computing the class-mean accuracy
    conf_matrix = confusion_matrix(true, pred)
    class_wise_acc = conf_matrix.diagonal()/conf_matrix.sum(1)
    mean_class_acc = np.nanmean(class_wise_acc) 
    class_wise_acc = {k:v for k, v in enumerate(class_wise_acc)}

    print(f'-> Top K accuracy :: {top_k_score}')
    print(f'-> Top 5 Class mean accuracy :: { mean_class_acc }')
    return

# =======================================================
# Evaluate model performance on dataset
# =======================================================

def validate(model, val_loader, val_dataset, args, train = True):

    true_action, true_verb, true_noun = [], [], [] 
    logit_action_list, logit_verb_list, logit_noun_list = [], [], []
    lm_feat_list = []
    uid_list = []

    model.eval()
    for idx, batch in enumerate(val_loader):

        max_seq_len = batch['attn_mask'].sum(dim = 1).max().item() + 3

        batch['input_id'] = batch['input_id'][:, :max_seq_len].to(torch.long).to(DEVICE)
        batch['attn_mask'] = batch['attn_mask'][:, :max_seq_len].to(torch.long).to(DEVICE)
        batch['label_action'] = batch['label_action'].to(torch.long).to(DEVICE)
        batch['label_verb'] = batch['label_verb'].to(torch.long).to(DEVICE)
        batch['label_noun'] = batch['label_noun'].to(torch.long).to(DEVICE)

        model_op = model(input_id = batch['input_id'], attn_mask = batch['attn_mask'])

        true_action.extend( batch['label_action'].cpu().numpy() )
        true_verb.extend( batch['label_action'].cpu().numpy() )
        true_noun.extend( batch['label_action'].cpu().numpy() )
        
        uid_list.extend( batch['uid'].numpy() )
        lm_feat_list.extend( model_op['LM_feat'].detach().cpu().numpy() )

        logit_action_list.extend( model_op['logit_action'].detach().cpu().numpy() )
        logit_verb_list.extend( model_op['logit_verb'].detach().cpu().numpy() )
        logit_noun_list.extend( model_op['logit_noun'].detach().cpu().numpy() )

    print(f'For Action :: ')
    compute_performance(true_action, logit_action_list, args)

    # print(f'\nFor Verb :: ')
    # compute_performance(true_verb, logit_verb_list, args)

    # print(f'\nFor Noun :: ')
    # compute_performance(true_noun, logit_noun_list, args)

    logit_action_list = np.array(logit_action_list)
    uid_list = np.array(uid_list)
    lm_feat_list = np.array(lm_feat_list)
    
    chpkt = '1MRecipe'
    if args.checkpoint_path == None:
        chpkt = 'None'
    if args.checkpoint_path == 'base':
        chpkt = 'base'

    # Only save the teacher prediction if Epoch >= 7
    if args.epoch >= 6:
        pkl_op_name = f'{args.model_type}_E_{args.epoch}_B_{args.batch_size}_T_{train}_W_{args.weigh_classes}_chpk_{chpkt}_MT_{args.multi_task}_hL_{args.hist_len}.pkl' 
        pickle_op = []

        for row in val_dataset.full_dataset:
            uid = row['uid']
            logit_action = logit_action_list[uid_list == uid]
            lm_feat = lm_feat_list[uid_list == uid]

            logit_action = logit_action.mean(0)
            lm_feat = lm_feat.mean(0)
            row['logit_action'] = logit_action[:, None]
            row['LM_feat'] = lm_feat[:, None]
            pickle_op.append(row)
        
        file_pi = open(os.path.join('./teacher_pred/epic_55_5', pkl_op_name), 'wb') 
        pickle.dump(pickle_op, file_pi)
        file_pi.close()
    return

def parse():
    parser = argparse.ArgumentParser(description="ACTION ANTICIAPTION - TEACHER (LM) MODEL !!")
    parser.add_argument('-model_type',default='bert', help='BERT or RoBERTa or other models')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=5, help='No. of epochs')
    parser.add_argument('-max_len', type=int, default=512, help='Max token length for the model')
    parser.add_argument('-hist_len', type=int, default=5, help='Max length of the history for a segment')
    parser.add_argument('-checkpoint_path', type=str, help='Path to the pretrained LM model')
    parser.add_argument('-weigh_classes', type=str, default='False', help='Weigh the CE loss by class count')
    parser.add_argument('-multi_task', type=str, default='True', help='Additional verb + noun loss')
    parser.add_argument('-no_action_classes', type=int, default=2513, help='No. action classes')
    parser.add_argument('-no_verb_classes', type=int, default=125, help='No. verb classes')
    parser.add_argument('-no_noun_classes', type=int, default=352, help='No. noun classes')
    parser.add_argument('-config_path', default='bert-base-uncased', type=str, help='No. noun classes')
    args = parser.parse_args()    
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    args = parse()

    args.weigh_classes = args.weigh_classes == 'True'    
    args.multi_task = args.multi_task == 'True'    
 
    if args.model_type == 'bert':
        args.config_path = "bert-base-uncased"
    if args.model_type == 'roberta':
        args.config_path = "roberta-base"
    if args.model_type == 'distillbert':
        args.config_path = "distilbert-base-uncased"
    if args.model_type == 'alberta':
        args.config_path = "albert-base-v2"
    if args.model_type == 'deberta':
        args.config_path = "microsoft/deberta-base"
    if args.model_type == 'electra':
        args.config_path = 'google/electra-base-discriminator' 

    print(f'ARGS :: {args}')

    #===============================================================================================================
    # Input is sequence of segment label for each video = [17, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18] and so on
    # Expected output: [[17], [17, 0], [17, 0, 1], ... [17, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]] and so on
    #===============================================================================================================

    # reading the lable to action mapping file
    id_2_label_mapping = pd.read_csv('./data/epic55_action_seq/actions.csv')
    id_2_label_mapping['verb_txt'] = id_2_label_mapping.action.apply(lambda x: x.split('_')[0])
    id_2_label_mapping['noun_txt'] = id_2_label_mapping.action.apply(lambda x: x.split('_')[1])
    id_2_label_mapping['noun_txt'] = id_2_label_mapping.noun_txt.apply(lambda x: " ".join(x.split(':')[-1::-1]))
    id_2_label_mapping['action'] = id_2_label_mapping.apply(lambda x: x['verb_txt'] + ' <v> ' + x['noun_txt'], axis = 1)
    id_2_label_mapping = {row['id']: {'action_txt': row['action'], 'noun_class': row['noun'], 'verb_class': row['verb']} for ix, row in id_2_label_mapping.iterrows()}
    
    many_shot_verbs = pd.read_csv('./data/epic55_action_seq/EPIC_many_shot_verbs.csv')
    many_shot_verbs = many_shot_verbs.verb_class.tolist()
    many_shot_nouns = pd.read_csv('./data/epic55_action_seq/EPIC_many_shot_nouns.csv')
    many_shot_nouns = many_shot_nouns.noun_class.tolist()
    many_shot_action_class_list = [k for k, v in id_2_label_mapping.items() if (v['verb_class'] in many_shot_verbs) and (v['noun_class'] in many_shot_nouns) ]

    tr_data = epic55_dataset(act_id2txt_mapping = id_2_label_mapping, 
                                data_path = './data/epic55_action_seq/action_segments_train_ek55_complete_hist_with_uid.pickle', args = args)

    val_data = epic55_dataset(act_id2txt_mapping = id_2_label_mapping, 
                                data_path = './data/epic55_action_seq/action_segments_val_ek55_complete_hist_with_uid.pickle', args = args)

    epic55_DL_train = DataLoader(tr_data, batch_size = args.batch_size, shuffle=False)
    epic55_DL_test = DataLoader(val_data, batch_size = args.batch_size, shuffle=False)

    # declaring the model
    model = egtea_model(args)
    model.to(DEVICE)

    # ==========================================================
    # Defining the optimizer & Model
    # ==========================================================

    no_decay = ['bias' , 'gamma', 'beta']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-7},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

    CE_loss_act = nn.CrossEntropyLoss()
    CE_loss_verb = nn.CrossEntropyLoss()
    CE_loss_noun = nn.CrossEntropyLoss()
    CE_normal = nn.CrossEntropyLoss()

    # ==========================================================
    # TRAINING LOOP
    # ==========================================================
    
    time_start = time.time()

    print(f'\n#############################   Testing performance   ################################')
    args.epoch = -1
    print(f'>> Testing performance at epoch {-1}')
    validate(model = model, val_loader = epic55_DL_test, args=args, train = False, val_dataset=val_data)
    print('########################################################################################\n')


    for epoch in range(args.num_epochs):
        args.epoch = epoch
        model.train()

        print(f'\n==========================>>>   EPOCH NO ::{epoch }  <<<===========================')
        # for idx, batch in tqdm(enumerate(epic55_DL_train), total = len(epic55_DL_train)):
        for idx, batch in enumerate(epic55_DL_train):

            max_seq_len = batch['attn_mask'].sum(dim = 1).max().item() + 3
            optimizer.zero_grad()

            batch['input_id'] = batch['input_id'][:, :max_seq_len].to(torch.long).to(DEVICE)
            batch['attn_mask'] = batch['attn_mask'][:, :max_seq_len].to(torch.long).to(DEVICE)
            batch['label_action'] = batch['label_action'].to(torch.long).to(DEVICE)
            batch['label_verb'] = batch['label_verb'].to(torch.long).to(DEVICE)
            batch['label_noun'] = batch['label_noun'].to(torch.long).to(DEVICE)
            
            model_op = model(input_id = batch['input_id'], attn_mask = batch['attn_mask'])

            loss_action = CE_loss_act(model_op['logit_action'], batch['label_action'])

            loss_verb = CE_loss_verb(model_op['logit_verb'], batch['label_verb'])
            loss_noun = CE_loss_noun(model_op['logit_noun'], batch['label_noun'])

            if args.multi_task == False:
                loss = 1*loss_action

            if args.multi_task == True:
                loss = 0.7*loss_action + 0.15*loss_verb + 0.15*loss_noun

            loss.backward()
            optimizer.step()
            
        print(f'\n############################   Training performance   ################################')
        print(f'>> Training performance at epoch {epoch}')
        validate(model = model, val_loader = epic55_DL_train, args=args, train = True, val_dataset=tr_data)
        print('########################################################################################\n')

        print(f'\n#############################   Testing performance   ################################')
        print(f'>> Testing performance at epoch {epoch}')
        validate(model = model, val_loader = epic55_DL_test, args=args, train = False, val_dataset=val_data)
        print('########################################################################################\n')

    time_end = time.time()

    print(f'\nTotal time taken :: {(time_end - time_start)}')
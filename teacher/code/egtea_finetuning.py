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

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# =======================================================
# Defining the teacher model for EGTEA++ dataset
# =======================================================

class egtea_model(nn.Module):
    def __init__(self, args):
        super().__init__()
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
        
        self.out_action = nn.Linear(768, args.no_action_classes)
        self.out_verb = nn.Linear(768, args.no_verb_classes)
        self.out_noun = nn.Linear(768, args.no_noun_classes)

    def forward(self, input_id, attn_mask):
        LM_op = self.LM_base(input_ids = input_id, attention_mask = attn_mask, return_dict = True) 
        LM_op_cls = LM_op['last_hidden_state'][:, 0, :]
        # LM_op = self.LM_base(input_ids = input_id, attention_mask = attn_mask) 
        # LM_op_cls = LM_op['pooler_output']
        output_action = self.out_action(LM_op_cls)    # dim --> (B, C)
        output_verb = self.out_verb(LM_op_cls)        # dim --> (B, C)
        output_noun = self.out_noun(LM_op_cls)        # dim --> (B, C)

        return {'logit_action': output_action, 'logit_verb': output_verb, 
                'logit_noun': output_noun, 'LM_feat': LM_op_cls}

# =======================================================
# Sorting the video segments dataset by Video_id -> start_time
# =======================================================
def sort_df(df, meta_data, args, train = True):

    if train == True:
        meta_data = meta_data[['uid', 'video_id', 'start_frame_30fps', 'end_frame_30fps', 'verb_class', 'noun_class', 'action_class']]
        meta_data.columns = ['uid', 'video_id', 'start', 'end', 'verb_class', 'noun_class', 'action_class']
    else:
        meta_data = meta_data[['uid', 'video_id', 'start', 'end', 'verb_class', 'noun_class', 'action_class']]

    full_DF = df.merge(meta_data, on = 'uid', how = 'left')
    
    if args.sort_seg == True:
        full_DF = full_DF.sort_values(['video_id', 'start'])
        
    return full_DF.to_dict('record')

# =======================================================
# Defining the 50 salad dataset
# =======================================================

class egtea_dataset(Dataset):
    def __init__(self, act_id2txt_mapping, data_path, meta_data, args, train = True):

        # reading the lable sequence file
        with open(data_path, "rb") as input_file:
            self.full_dataset = pickle.load(input_file)

        # If the query segments are to be sorted by their time-stamp
        self.full_dataset = sort_df(df = pd.DataFrame(self.full_dataset), 
                                    meta_data = meta_data, train = train, args = args)
        
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
            for segment_seq in seg_history_list:

                # Only use the last K segments of the history sequence
                ip_txt = [act_id2txt_mapping[id] for id in segment_seq[(-args.hist_len)::1]]
                ip_txt = " <a> ".join(ip_txt)

                self.ip_txt_list.append(ip_txt)
                self.op_label_action.append( seg_dict['action_class'] )
                self.op_label_verb.append( seg_dict['verb_class'] )
                self.op_label_noun.append( seg_dict['noun_class'] )
                self.uid.append( seg_dict['uid'] )

                assert seg_dict['action_class'] == seg_dict['target']

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

def compute_performance(true, pred_logit):

    top_k_score = {}
    pred = [p.argmax() for p in pred_logit]

    # Computing the top 5 accuracy
    for k in range(1, 6):
        topK_acc = sum([g in np.argsort(-p)[:k] for p, g in zip(pred_logit, true)])/len(true)
        top_k_score[k] = topK_acc

    # Computing the class-mean accuracy
    conf_matrix = confusion_matrix(true, pred)
    class_wise_acc = conf_matrix.diagonal()/conf_matrix.sum(1)
    mean_class_acc = np.nanmean(class_wise_acc) 
    class_wise_acc = {k:v for k, v in enumerate(class_wise_acc)}

    print(f'-> Top K accuracy :: {top_k_score}')
    print(f'-> Class mean accuracy :: { mean_class_acc }')
    # print(f'-> Class wise accuracy :: \n{ class_wise_acc }')
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
    soft_max = torch.nn.Softmax(dim = 1)

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

    compute_performance(true_action, logit_action_list)

    logit_action_list = np.array(logit_action_list)
    uid_list = np.array(uid_list)
    lm_feat_list = np.array(lm_feat_list)
    
    chpkt = '1MRecipe'
    if args.checkpoint_path == None:
        chpkt = 'None'
    if args.checkpoint_path == 'base':
        chpkt = 'base'

    if args.epoch >= 2:
        pkl_op_name = f'{args.model_type}_E_{args.epoch}_T_{train}_W_{args.weigh_classes}_chpk_{chpkt}_MT_{args.multi_task}_hL_{args.hist_len}_SS_{args.sort_seg}_GS_{args.gappy_hist}.pkl' 
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
        
        file_pi = open(os.path.join('./teacher_pred/egtea_2', pkl_op_name), 'wb') 
        pickle.dump(pickle_op, file_pi)
        file_pi.close()
    return

def parse():
    parser = argparse.ArgumentParser(description="ACTION ANTICIAPTION - TEACHER (LM) MODEL !!")
    parser.add_argument('-model_type',default='bert', help='BERT or RoBERTa')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=8, help='No. of epochs')
    parser.add_argument('-max_len', type=int, default=512, help='Max token length for the model')
    parser.add_argument('-hist_len', type=int, default=25, help='Max length of the history for a segment')
    parser.add_argument('-checkpoint_path', type=str, help='Path to the pretrained LM model')
    parser.add_argument('-weigh_classes', type=str, default='False', help='Weigh the CE loss by class count')
    parser.add_argument('-multi_task', type=str, default='False', help='Additional verb + noun loss')
    parser.add_argument('-sort_seg', type=str, default='True', help='Sort the segments by time-stamp?')
    parser.add_argument('-gappy_hist', type=str, default='True', help='Is there gap in the history?')
    parser.add_argument('-no_action_classes', type=int, default=106, help='No. action classes')
    parser.add_argument('-no_verb_classes', type=int, default=19, help='No. verb classes')
    parser.add_argument('-no_noun_classes', type=int, default=51, help='No. noun classes')
    parser.add_argument('-config_path', default='bert-base-uncased', type=str, help='Config path for the model')
    args = parser.parse_args()    
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    args = parse()
    args.weigh_classes = args.weigh_classes == 'True'    
    args.multi_task = args.multi_task == 'True'    
    args.sort_seg = args.sort_seg == 'True'
    args.gappy_hist = args.gappy_hist == 'True'

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
    with open('./data/egtea_action_seq/actions.csv', "rb+") as input_file:
        action_dataset = input_file.read()
        
    action_dataset = str(action_dataset,'utf-8').splitlines()
    action_dataset = [act.split(',') for act in action_dataset]
    action_dataset = [[int(act[0]), act[-1].strip().lower()] for act in action_dataset]

    id2txt_mapping = [act[-1] for act in action_dataset]
    id2txt_mapping = list(map(lambda x: x.replace('_', ' <v> '), id2txt_mapping))
    id2txt_mapping = list(map(lambda x: x.replace('/', ' or '), id2txt_mapping))
    id2txt_mapping = list(map(lambda x: x.replace(':', ' '), id2txt_mapping))
    id2txt_mapping = {k:v for k, v in enumerate(id2txt_mapping)}    
        
    tr_metadata = pd.read_csv('./data/egtea_action_seq/training1.csv')
    val_metadata = pd.read_csv('./data/egtea_action_seq/validation1.csv')
    
    if args.gappy_hist == True:
        tr_data = egtea_dataset(act_id2txt_mapping = id2txt_mapping, meta_data= tr_metadata,
                                data_path= './data/egtea_action_seq/action_segments_egtea.pickle',
                                args = args, train=True)
        val_data = egtea_dataset(act_id2txt_mapping = id2txt_mapping, meta_data= val_metadata,
                                data_path= './data/egtea_action_seq/val_action_segments_egtea.pickle',
                                args = args, train=False)
    else:
        tr_data = egtea_dataset(act_id2txt_mapping = id2txt_mapping, meta_data= tr_metadata,
                                data_path= './data/egtea_action_seq/action_segments_egtea_complete_hist.pickle',
                                args = args, train=True)
        val_data = egtea_dataset(act_id2txt_mapping = id2txt_mapping, meta_data= val_metadata,
                                data_path= './data/egtea_action_seq/val_action_segments_egtea_complete_hist.pickle',
                                args = args, train=False)

    egtea_DL_train = DataLoader(tr_data, batch_size = args.batch_size, shuffle=True)
    egtea_DL_test = DataLoader(val_data, batch_size = args.batch_size, shuffle=True)

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

    if args.weigh_classes == True:

        tr_action_count = np.array([ele['action_class'] for ele in tr_data.full_dataset])
        tr_verb_count = np.array([ele['verb_class'] for ele in tr_data.full_dataset])
        tr_noun_count = np.array([ele['noun_class'] for ele in tr_data.full_dataset])

        action_class_weights = cls_weigh('balanced', np.unique(tr_action_count).tolist(), tr_action_count.tolist())
        verb_class_weights = cls_weigh('balanced', np.unique(tr_verb_count).tolist(), tr_verb_count.tolist())
        noun_class_weights = cls_weigh('balanced', np.unique(tr_noun_count).tolist(), tr_noun_count.tolist())

        CE_loss_act = nn.CrossEntropyLoss( weight = torch.tensor(action_class_weights).to(torch.float).to(DEVICE) )
        CE_loss_verb = nn.CrossEntropyLoss( weight = torch.tensor(verb_class_weights).to(torch.float).to(DEVICE) )
        CE_loss_noun = nn.CrossEntropyLoss( weight = torch.tensor(noun_class_weights).to(torch.float).to(DEVICE) )

    # ==========================================================
    # TRAINING LOOP
    # ==========================================================
    
    time_start = time.time()
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        model.train()

        print(f'\n==========================>>>   EPOCH NO ::{epoch }  <<<===========================')
        # for idx, batch in tqdm(enumerate(egtea_DL_train), total = len(egtea_DL_train)):
        for idx, batch in enumerate(egtea_DL_train):

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
                loss = 0.8*loss_action + 0.1*loss_verb + 0.1*loss_noun

            loss.backward()
            optimizer.step()
            
        print(f'\n############################   Training performance   ################################')
        print(f'>> Training performance at epoch {epoch}')
        validate(model = model, val_loader = egtea_DL_train, args=args, train = True, val_dataset=tr_data)
        print('########################################################################################\n')

        print(f'\n#############################   Testing performance   ################################')
        print(f'>> Testing performance at epoch {epoch}')
        validate(model = model, val_loader = egtea_DL_test, args=args, train = False, val_dataset=val_data)
        print('########################################################################################\n')

    time_end = time.time()

    print(f'\nTotal time taken :: {(time_end - time_start)}')
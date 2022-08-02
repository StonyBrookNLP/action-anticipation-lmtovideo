import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, AdamW, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
import torch
import os
from copy import deepcopy as cc
import pdb
from tqdm import tqdm
import torch.nn as nn
import pickle
import argparse

# =======================================================
# Defining the teacher model for 50 salad dataset
# =======================================================

class BERT_text_classification(nn.Module):
    def __init__(self, checkpoint_path = None, no_classes = 19, model_type = 'bert'):
        super().__init__()

        if model_type == 'roberta':
            print('>> Model type is :: {RoBERTa}')
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
            if checkpoint_path != None:
                self.LM_base = RobertaModel.from_pretrained(checkpoint_path)
            else:
                self.LM_base = RobertaModel.from_pretrained('roberta-base')
        else:
            print('>> Model type is :: {BERT}')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
            if checkpoint_path != None:
                self.LM_base = BertModel.from_pretrained(checkpoint_path)
            else:
                self.LM_base = BertModel.from_pretrained('bert-base-uncased')

        self.out = nn.Linear(768, no_classes)

    def forward(self, input_id, attn_mask):
        
        LM_op = self.LM_base(input_ids = input_id, attention_mask = attn_mask) 
        LM_op_cls = LM_op['pooler_output']
        output = self.out(LM_op_cls)        # dim --> (B, C)
        return {'logit': output, 'LM_feat': LM_op_cls}

# =======================================================
# Defining the 50 salad dataset
# =======================================================

class _50salad_dataset(Dataset):
    def __init__(self, label_seq_idx, id2txt_mapping, max_length = 512, train = True, model_type = 'bert'):

        self.train = train

        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.id_2_verb_mapping = {k:v.split('_')[0] for k, v in id2txt_mapping.items()}
        self.id_2_noun_mapping = {k: " ".join(v.split('_')[1:]) for k,v in id2txt_mapping.items()}

        ip_txt = []
        ip_id = []
        self.op_label = []

        for ix in range(len(label_seq_idx)):
            label_sequence = label_seq_idx[ix]
            ip_seq = cc(label_sequence[:-1]) if self.train == True else cc(label_sequence)
            
            # Create sequence of token (string) from the sequence of labels
            ip_seq_txt = []
            for id in ip_seq:
                action = self.id_2_verb_mapping[id] + ' <v> ' + self.id_2_noun_mapping[id]
                ip_seq_txt.append(action)
            ip_seq_txt = " <a> ".join(ip_seq_txt)

            ip_txt.append(ip_seq_txt)

            self.op_label.append(cc(label_sequence[-1]))

        # Encoding the text
        self.batch_encodings = self.tokenizer.batch_encode_plus(ip_txt, max_length = max_length, padding = True, truncation = True)
        self.batch_encodings.input_ids = torch.tensor(self.batch_encodings.input_ids)
        self.batch_encodings.attention_mask = torch.tensor(self.batch_encodings.attention_mask)
        self.op_label = torch.tensor( self.op_label)

    def __len__(self):
        return len(self.batch_encodings.input_ids)

    def __getitem__(self, idx):
        if self.train == False:
            return {'input_id': self.batch_encodings.input_ids[idx], 'attn_mask' : self.batch_encodings.attention_mask[idx]}
        else:
            return {'input_id': self.batch_encodings.input_ids[idx], 
                    'attn_mask' : self.batch_encodings.attention_mask[idx], 
                    'label': self.op_label[idx]}

def validate(model, val_loader, top_K = 5):

        g_truth = []
        pred = []
        pred_topk = []
        model.eval()
        soft_max = torch.nn.Softmax(dim = 1)

        for idx, batch in enumerate(val_loader):

                max_seq_len = batch['attn_mask'].sum(dim = 1).max().item() + 3

                batch['input_id'] = batch['input_id'][:, :max_seq_len].to(torch.long).to(DEVICE)
                batch['attn_mask'] = batch['attn_mask'][:, :max_seq_len].to(torch.long).to(DEVICE)
                batch['label'] = batch['label'].to(torch.long).to(DEVICE)

                model_op = model(input_id = batch['input_id'], attn_mask = batch['attn_mask'])
                
                pred_topk.extend(soft_max(model_op['logit']).detach().cpu().numpy() )
                g_truth.extend(batch['label'].cpu().numpy())
                pred.extend(model_op['logit'].argmax(dim = 1).detach().cpu().numpy())

        top1_acc = sum([p == g for p, g in zip(pred, g_truth)])/len(g_truth)
        topK_acc = sum([g in np.argsort(-p)[:top_K] for p, g in zip(pred_topk, g_truth)])/len(g_truth)

        print(f'Top 1 Accuracy :: {top1_acc}')
        print(f'Top {top_K} Accuracy :: {topK_acc}')
        return top1_acc, topK_acc

def infer(model, dat_loader):

    model.eval()
    pred = []
    LM_feat = []
    for idx, batch in enumerate(dat_loader):
            max_seq_len = batch['attn_mask'].sum(dim = 1).max().item() + 3
            batch['input_id'] = batch['input_id'][:, :max_seq_len].to(torch.long).to(DEVICE)
            batch['attn_mask'] = batch['attn_mask'][:, :max_seq_len].to(torch.long).to(DEVICE)

            model_op = model(input_id = batch['input_id'], attn_mask = batch['attn_mask'])            
            pred.extend(model_op['logit'].detach().cpu().numpy())
            LM_feat.extend(model_op['LM_feat'].detach().cpu().numpy())

    return pred, LM_feat

def parse():
    parser = argparse.ArgumentParser(description="ACTION ANTICIAPTION - TEACHER (LM) MODEL !!")
    parser.add_argument('-model_type',default='bert', help='BERT or RoBERTa')
    parser.add_argument('-batch_size', type=int, default=12, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=8, help='No. of epochs')
    parser.add_argument('-out_path', type=str, help='Path of the output dictionary')
    parser.add_argument('-pre_trained_LM_path', type=str, help='Path to the pretrained LM model')
    parser.add_argument('-shuffle_tr_batch', type=str, default='True', help='Shuffle training batch')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    No_folds = 5

    args = parse()
    print(f'>> Arguments :: {args}')

    BATCH_SIZE = args.batch_size
    NO_EPOCHS = args.num_epochs
    model_type = args.model_type
    pre_trained_LM_path = args.pre_trained_LM_path
    model_pretrained = pre_trained_LM_path != None

    shuffle_tr_batch = args.shuffle_tr_batch == 'True'

    # Model out naming convention
    # m: model type
    # e: no. epochs
    # b: batch size
    # s: batch shuffled
    # p: pre-trained model loaded or not

    out_path = f'm_{model_type}_e_{NO_EPOCHS}_b_{BATCH_SIZE}_s_{shuffle_tr_batch}_p_{model_pretrained}.pkl'
    out_path = os.path.join('./teacher_pred/50_salad_temp', out_path)

    #===============================================================================================================
    # Input is sequence of segment label for each video = [17, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18] and so on
    # Expected output: [[17], [17, 0], [17, 0, 1], ... [17, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]] and so on
    #===============================================================================================================

    # reading the lable sequence file
    with open('./data/50_salad_action_seq/sequences_with_fold.pickle', "rb") as input_file:
        label_seq_idx_with_split = pickle.load(input_file)

    # only consider label sequence with length >= 2
    label_seq_idx = [seq['sequence'][:s] for seq in label_seq_idx_with_split for s in range(1, len(seq['sequence']))]
    label_seq_idx_full = cc(label_seq_idx)                              # will be used during inference
    label_seq_idx = [seq for seq in label_seq_idx if len(seq) >1]       # will be used for training ; end of each sequence is the target, so the seq length should be >= 1

    split_id =  [seq['split_no'] for seq in label_seq_idx_with_split for s in range(1, len(seq['sequence']))]
    split_id_full =  cc(split_id)
    split_id = [idx for idx, seq in zip(split_id, label_seq_idx) if len(seq) >1]

    label_seq_idx_tr_kfold = []
    label_seq_idx_te_kfold = []

    for j in range(1, (No_folds + 1)):

        tr_dat_to_be_added = [label_seq_idx[i] for i in range(len(label_seq_idx)) if split_id[i] != j]
        te_dat_to_be_added = [label_seq_idx[i] for i in range(len(label_seq_idx)) if split_id[i] == j]

        label_seq_idx_tr_kfold.append( tr_dat_to_be_added )
        label_seq_idx_te_kfold.append( te_dat_to_be_added )

    # reading the lable to action mapping file
    with open('data/50_salad_action_seq/mapping_new.txt', "rb+") as input_file:
        id2txt_mapping = input_file.read()
    id2txt_mapping = str(id2txt_mapping,'utf-8')
    id2txt_mapping = {int(v.split()[0]) : str(v.split()[1]) for v in id2txt_mapping.splitlines()}

    dataset_50salad_full = _50salad_dataset(label_seq_idx = label_seq_idx_full, 
                                    id2txt_mapping = id2txt_mapping, 
                                    max_length = 512, 
                                    train = False)

    recipe_loader_full = DataLoader(dataset_50salad_full, batch_size = BATCH_SIZE, shuffle=False)
    inference = []

    top1_vl_acc_avg, top5_vl_acc_avg, top1_tr_acc_avg, top5_tr_acc_avg = 0, 0, 0, 0

    for fold in range(5):
        print(f'>>-------------------------Fold no :: {fold} -------------------------<<')

        dataset_50salad = _50salad_dataset(label_seq_idx = label_seq_idx_tr_kfold[fold], 
                                            id2txt_mapping = id2txt_mapping, 
                                            max_length = 512, 
                                            train = True)
                                            
        dataset_50salad_val = _50salad_dataset(label_seq_idx = label_seq_idx_te_kfold[fold], 
                                            id2txt_mapping = id2txt_mapping, 
                                            max_length = 512, 
                                            train = True)

        recipe_loader = DataLoader(dataset_50salad, batch_size = BATCH_SIZE, shuffle = shuffle_tr_batch)
        recipe_loader_val = DataLoader(dataset_50salad_val, batch_size = BATCH_SIZE, shuffle=True)

        # definining the LM based model
        model = BERT_text_classification(checkpoint_path = pre_trained_LM_path, no_classes = 19, 
                                        model_type = model_type)
        model.to(DEVICE)

        # ==========================================================
        # Defining the optimizer & Model
        # ==========================================================

        no_decay = ['bias' , 'gamma', 'beta']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
        CE_loss = nn.CrossEntropyLoss(ignore_index = -100)

        # ==========================================================
        # TRAINING LOOP
        # ==========================================================
        tr_loss = []
        for epoch in tqdm(range(NO_EPOCHS), total=NO_EPOCHS):
        #for epoch in range(NO_EPOCHS):
            model.train()
            # for idx, batch in tqdm(enumerate(recipe_loader), total = len(recipe_loader)):
            for idx, batch in enumerate(recipe_loader):

                max_seq_len = batch['attn_mask'].sum(dim = 1).max().item() + 3
            
                optimizer.zero_grad()

                batch['input_id'] = batch['input_id'][:, :max_seq_len].to(torch.long).to(DEVICE)
                batch['attn_mask'] = batch['attn_mask'][:, :max_seq_len].to(torch.long).to(DEVICE)
                batch['label'] = batch['label'].to(torch.long).to(DEVICE)
                
                model_op = model(input_id = batch['input_id'], attn_mask = batch['attn_mask'])
                loss = CE_loss(model_op['logit'], batch['label'])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            print(f'>> Training performance at epoch :: {epoch}')
            _, _ = validate(model = model, val_loader = recipe_loader)
            
            print(f'>> Testing performance at epoch :: {epoch}')
            _, _ = validate(model = model, val_loader = recipe_loader_val)


        print(f'>> Training performance at epoch :: {epoch}')
        top1_tr_acc, top5_tr_acc = validate(model = model, val_loader = recipe_loader)
        
        print(f'>> Testing performance at epoch :: {epoch}')
        top1_vl_acc, top5_vl_acc = validate(model = model, val_loader = recipe_loader_val)

        top1_tr_acc_avg += top1_tr_acc 
        top5_tr_acc_avg += top5_tr_acc
        top1_vl_acc_avg += top1_vl_acc 
        top5_vl_acc_avg += top5_vl_acc

        print(f'\n= ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ \n')

        model_pred, LM_feat_list = infer(model, recipe_loader_full)

        inference_for_fold = []
        for ip, pred, LM_feat in zip(label_seq_idx_full, model_pred, LM_feat_list):
            element_to_be_added = {'fold_no': fold, 'sequence': ip, 
                                   'LM_pred': pred, 'LM_feat': LM_feat, 
                                   'model_type': model_type, 'pre_trained': pre_trained_LM_path != None}
            inference_for_fold.append(element_to_be_added)

        inference.extend(inference_for_fold)

    top1_tr_acc_avg /= 5.0 
    top5_tr_acc_avg /= 5.0
    top1_vl_acc_avg /= 5.0 
    top5_vl_acc_avg /= 5.0
    print(f'Top 1 training :: {top1_tr_acc_avg} Top 5 training :: {top5_tr_acc_avg} Top 1 validation :: {top1_vl_acc_avg} Top 5 validation:: {top5_vl_acc_avg}')

    file_pi = open(out_path, 'wb') 
    pickle.dump(inference, file_pi)
    file_pi.close()

    with open(out_path, "rb") as input_file:
        temp = pickle.load(input_file)

    print(len(temp))
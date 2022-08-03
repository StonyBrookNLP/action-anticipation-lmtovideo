
import pickle
import torch


def process_lm_pred(lm_pred):
    lm_pred_dict = {}
    lm_pred_feat_dict = {}
    for pred in lm_pred:
        lm_pred_dict[pred['uid']] = torch.from_numpy(pred['logit_action']).squeeze()
        # lm_pred_feat_dict[pred['uid']] = torch.from_numpy(pred['LM_feat']).squeeze()

    return lm_pred_dict, lm_pred_feat_dict


if __name__ == '__main__':
    fnames = ['/Users/tanviaggarwal/Desktop/SOP/model_ek55/not_pretrained_LM_preds/bert_E_9_B_16_T_True_W_False_chpk_base_MT_True_hL_5.pkl',
              '/Users/tanviaggarwal/Desktop/SOP/model_ek55/not_pretrained_LM_preds/alberta_E_9_B_16_T_True_W_False_chpk_base_MT_True_hL_5.pkl',
              '/Users/tanviaggarwal/Desktop/SOP/model_ek55/not_pretrained_LM_preds/distillbert_E_9_B_16_T_True_W_False_chpk_base_MT_True_hL_5.pkl',
              '/Users/tanviaggarwal/Desktop/SOP/model_ek55/not_pretrained_LM_preds/roberta_E_9_B_16_T_True_W_False_chpk_base_MT_True_hL_5.pkl',
              '/Users/tanviaggarwal/Desktop/SOP/model_ek55/not_pretrained_LM_preds/electra_E_9_B_16_T_True_W_False_chpk_base_MT_True_hL_5.pkl']
    pred_fnames = ['./data/lm_pred_ek55_bert_may13.pickle',
                   './data/lm_pred_ek55_alberta_may13.pickle',
                   './data/lm_pred_ek55_distillbert_may13.pickle',
                   './data/lm_pred_ek55_roberta_may13.pickle',
                   './data/lm_pred_ek55_electra_may13.pickle']
    for i in range(len(fnames)):
        with open(fnames[i], 'rb') as handle:
            lm_pred_bert = pickle.load(handle)
        print(len(lm_pred_bert))

        pred_dict, feat_dict = process_lm_pred(lm_pred_bert)

        with open(pred_fnames[i], 'wb') as handle:
            pickle.dump(pred_dict, handle)
    # with open(f'./lm_pred_feat_ek55_16_apr.pickle', 'wb') as handle:
    #     pickle.dump(feat_dict, handle)

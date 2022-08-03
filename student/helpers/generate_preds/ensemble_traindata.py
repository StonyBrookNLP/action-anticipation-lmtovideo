
import pickle
import torch


def process_lm_pred(lm_pred):
    lm_pred_dict = {}
    lm_pred_feat_dict = {}
    lm_mean_pred_dict = {}
    lm_maxpool_pred_dict = {}
    for pred in lm_pred:
        lm_pred_dict[pred['uid']] = torch.from_numpy(pred['logit_action']).squeeze()
        lm_pred_feat_dict[pred['uid']] = torch.from_numpy(pred['LM_feat']).squeeze()
        lm_mean_pred_dict[pred['uid']] = torch.from_numpy(pred['logit_action_mean']).squeeze()
        lm_maxpool_pred_dict[pred['uid']] = torch.from_numpy(pred['logit_action_maxpool']).squeeze()
    return lm_pred_dict, lm_pred_feat_dict, lm_mean_pred_dict, lm_maxpool_pred_dict


if __name__ == '__main__':
    with open('/Users/tanviaggarwal/Desktop/SOP/model_ek55/not_pretrained_LM_preds/Ensemble_E_9_B_16_T_True_W_False_chpk_base_MT_True_hL_5.pkl', 'rb') as handle:
        lm_pred_bert = pickle.load(handle)
    print(len(lm_pred_bert))

    pred_dict, feat_dict, mean_dict, maxpool_dict = process_lm_pred(lm_pred_bert)

    # with open(f'./data/lm_pred_ensemble_ek55_6may.pickle', 'wb') as handle:
    #     pickle.dump(pred_dict, handle)
    # with open(f'./data/lm_pred_feat_ensemble_ek55_6may.pickle', 'wb') as handle:
    #     pickle.dump(feat_dict, handle)
    # with open(f'./data/lm_pred_mean_ensemble_not_pretrained_ek55_13may.pickle', 'wb') as handle:
    #     pickle.dump(mean_dict, handle)
    with open(f'./data/lm_pred_maxpool_ensemble_not_pretrained_ek55_13may.pickle', 'wb') as handle:
        pickle.dump(mean_dict, handle)

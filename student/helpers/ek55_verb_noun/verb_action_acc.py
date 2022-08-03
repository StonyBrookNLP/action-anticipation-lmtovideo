import numpy as np
import pandas as pd
import pickle


def get_map_to_action_class(fname):
    df = pd.read_csv(fname)
    action_verb_noun_map = {}
    for action, action_row in df.iterrows():
        verb, noun = action_row.verb, action_row.noun
        action_verb_noun_map[int(action_row.id)] = [int(verb), int(noun)]
    return action_verb_noun_map


def decompose_action_list(action_map, actions):
    verbs = []
    nouns = []
    for action_list in actions:
        verb_list = []
        noun_list = []
        for action in action_list:
            verb_list.append(action_map[int(action)][0])
            noun_list.append(action_map[int(action)][1])
        verbs.append(verb_list)
        nouns.append(noun_list)
    return np.array(verbs), np.array(nouns)


def decompose_action(action_map, actions):
    verbs = []
    nouns = []
    for action in actions:
        verbs.append(action_map[int(action)][0])
        nouns.append(action_map[int(action)][1])
    return np.array(verbs), np.array(nouns)


# def get_acc(predicted, target):
#     n = len(target)
#     correct = 0
#     for i in range(n):
#         if predicted[i] == target[i]:
#             correct += 1
#     return correct / n


def get_acc(predictions, labels):
    ratio_solved = np.mean(
        np.any(labels[:, np.newaxis] == predictions, axis=-1))
    return ratio_solved * 100.0


def get_top_k_acc(action_verb_noun_map, preds, target, k):
    # given: target action (verb+noun)
    # predicted: action
    # logic:
    # decompose predicted action into predicted verb + predicted noun
    # compare target verb, predicted verb
    # compare target noun, predicted noun

    prediction_action = np.argpartition(preds, -k, axis=-1)[:, -k:]
    target_verb, target_noun = decompose_action(action_verb_noun_map, target.tolist())
    predicted_verb, predicted_noun = decompose_action_list(action_verb_noun_map, prediction_action)
    action_acc = get_acc(prediction_action, target)
    verb_acc = get_acc(predicted_verb, target_verb)
    noun_acc = get_acc(predicted_noun, target_noun)

    return action_acc, verb_acc, noun_acc


if __name__ == "__main__":
    action_map = get_map_to_action_class('../ek55_verb_noun/actions.csv')
    preds_paths = ['/Users/tanviaggarwal/Desktop/SOP/model_results/ek55/baseline1_preds.pickle',
                   '../ek55_preds/notpretrained_ens_attention_nh4_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/notpretrained_ens_mean_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_ens_attention_nh4_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_ens_mean_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/notpretrained_alberta_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/notpretrained_roberta_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/notpretrained_electra_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/notpretrained_bert_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/notpretrained_distillbert_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_alberta_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_roberta_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_electra_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_bert_kldiv_top50_wt20_t5_preds.pickle',
                   '../ek55_preds/pretrained_distillbert_kldiv_top50_wt20_t5_preds.pickle'
                   ]
    for pred_file in preds_paths:
        print("-------------")
        print(pred_file)
        print("-------------")
        with open(pred_file, 'rb') as handle:
            preds = pickle.load(handle)
        action_acc, verb_acc, noun_acc = get_top_k_acc(action_map, preds['logits'], preds['target'], 1)
        action_acc_5, verb_acc_5, noun_acc_5 = get_top_k_acc(action_map, preds['logits'], preds['target'], 5)
        print("Top1 action: {}\nTop5 action: {}".format(round(action_acc, 2), round(action_acc_5, 2)))
        print("Top1 verb: {}\nTop5 verb: {}".format(round(verb_acc, 2), round(verb_acc_5, 2)))
        print("Top1 noun: {}\nTop5 noun: {}".format(round(noun_acc, 2), round(noun_acc_5, 2)))
        # print("{}\t{}\t{}\t{}".format(round(verb_acc, 2), round(verb_acc_5, 2), round(noun_acc, 2), round(noun_acc_5, 2) ))
        print("-------------")

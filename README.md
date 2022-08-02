# Action_Anticipation

- ## Teacher training
  1. Copy the following directory /home/sayontan/Action_anticipation/
      ```
        data  
        └───50_salad_action_seq/
        └───egtea_action_seq/ 
        └───epic55_action_seq/
      ```
  2. To train EGTEA+ 
    ```
      python ./code/egtea_finetuning.py \
      -model_type bert \ # bert/roberta/distillbert/alberta/deberta/electra
      -batch_size 16 \   # batch-size
      -num_epochs 5 \    # no. of epochs
      -max_len 512 \ # Max # of tokens in the input, tokens beyond this number will be truncated
      -checkpoint_path ./out/bert_pretrained/checkpoint-200000 \ # path to the model checkpoint (for initialization) that was trained on 1M-Recipe through MLM
      -weigh_classes True/ #for imbalanced data, if True, then the loss will be weighted Cross-Entropy; will not work with EPIC-55 as few clases have 0 data instances
      -hist_len 15 \ # context length, i.e. how many actions in the past conditioned on which you want to predict the action after the anticipation time
      -gappy_hist True \ # EGTEA data (action sequence) have gaps as action segments in a video are partiioned into train/test set
      -multi_task True \ # Instead of just predicting the action, also predict the verb and noun
      -sort_seg True \ # Action segnment in the training batch should be sorted by their temporal order ?
    ```

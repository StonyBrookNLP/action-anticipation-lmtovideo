### [Text-Derived Knowledge Helps Vision: A Simple Cross-modal Distillation for Video-based Action Anticipation](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ns4fY0sAAAAJ&citation_for_view=ns4fY0sAAAAJ:UeHWp8X0CEIC)

- ## Teacher training
1. Create the environment using the <b>requirement.txt</b> file.
2. Download the pre-trained language models pretrained on the 1M dataset from the following location:
3.	Download the content of the <b>data</b> folder from the google drive [link](https://drive.google.com/drive/folders/1IA3G0YrXalotivQWQa4r5DHFylhJEe9K?usp=drive_link)
      ```
	      data
			└───egtea_action_seq/
			└───epic55_action_seq/
			└───recipe1M_layers/
			└───processed_data_dict.pt
			└───vocab.txt
      ```
4.	Download the content of <b>out</b> folder from the google drive [link](https://drive.google.com/drive/folders/1LXQ3sguMliFaZsRCdc2gq54Lm4oxktyo?usp=drive_link). 
      ```
	      out
			└───albert_pretrained/checkpoint-200000/
			└───bert_pretrained/checkpoint-200000/
			└───distilbert_pretrained/checkpoint-200000/
			└───electra_pretrained/checkpoint-200000/
			└───roberta_pretrained/checkpoint-200000/
			└─...
			.
      ```

5.	LM checkpoints pre-trained on action sequences derived from 1M recipe are in their respective folders
6. Code for pretraining LM on 1M-Recipe is at ```code/{model_name}_pretraining.py```

 7. To finetune model on EGTEA-GAZE+ dataset
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
      -sort_seg True \ # Action segnment in the training batch should be sorted by their temporal order 
    ```

 8. To finetune model on EPIC-55 dataset

	```
      python ./code/epic55_finetuning.py \
	      -model_type distillbert \
	      -batch_size  16 \
	      -num_epochs 8 \
	      -max_len 512 \
	      -hist_len 5 \
	      -checkpoint_path ./out/distilbert_pretrained/checkpoint-200000 \
	      -weigh_classes False \
	      -multi_task True
	 ```
    
  9.  For the Egtea and EPIC55 dataset, the arguments in the above snippet are the model hyperparameter used to perform the teacher training and reporting the performance.
10. Sample slurm script can be found in ```code/slurm scripts```
10. Model (teacher) predictions for the test data can be found at the google drive [link](https://drive.google.com/drive/folders/1RvnH8Bc_6lyuj4JIok_cGR1qJHrQV5TI?usp=drive_link). 

11. The predictions are saved as list of dictionary, where each element of the list has the following keys ```<UID (unique segment ID), action_logit, LM_feature>```along with other segment (UID) associated such as actionID, action history, etc.

12. These teacher predictions are then used to train student [```Anticipative Video Transformer```](https://github.com/facebookresearch/AVT), through knowledge distillation.

 13. <b>Reproducing teacher metrics reported in the paper: </b>Download the model prediction folder ```teacher/teacher_student_Predictions/``` from [link](https://drive.google.com/drive/folders/1RvnH8Bc_6lyuj4JIok_cGR1qJHrQV5TI?usp=drive_link). Calculate the teacher performance metric reported in the paper by running the notebook ```code/logit_analysis.ipynb```
**![](https://lh3.googleusercontent.com/p92t1BHLaVym2JM20eRcIyD1ezlHMiFgGF4-1ZhIlPt9roJLcvzWU1wZBQUVBYnK1Sr7BIT6G3dwXFBLeEfK0yYJr6FAmIhirPzPAa2u6j6yK2Y36RoLb5qfnuGElrTV9YVSuHEPlsXqWVGVqqGSstk)**


- ## Student training
1.  Student training repo is in ```student```.  Our student training code is adopted from the ```AVT``` code base ([link](https://github.com/facebookresearch/AVT)). As such the ```DATA``` and other ```AVT```  model checkpoints should be first downloaded as explained in their documentation.

2. Setup the ```avt.yml``` python environment.

3.  Since most of the codes are same as ```AVT``` so their documentation can be referred to for reference purposes, and we explain and describe where in the codebase these different additions we made.

4. For all the experiments, we use exactly the same hyperparameters that the ```AVT``` authors used to train their model.


5.  Helper codes
	
	 - Generating LM (teacher) pred logits/features 
		https://github.com/sayontang/Action_Anticipation/blob/main/student/helpers/generate_preds/traindata.py

     - Processing result files

		- Get logits from the student
		https://github.com/sayontang/Action_Anticipation/blob/main/student/helpers/metrics_multiple.py

		- Compute verb/noun accuracy from action accuracy https://github.com/sayontang/Action_Anticipation/tree/main/student/helpers/egtea_verb_noun
		https://github.com/sayontang/Action_Anticipation/tree/main/student/helpers/ek55_verb_noun

6.  Config description
	 - Set weight for different losses (teacher + student during student training)
	 https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/conf/config.yaml#L61
**![](https://lh4.googleusercontent.com/bEJddNv23soUXTGKvCeiPjMU4_ZlUaJGG_nDZu2WroFJaVtnzzm35zt0RkAG0qiyU6CcjgyAn6txoaS0cxhRS2LBZHzZwPUo34PH0Q_29VfGhqZlBBJTsChtbCKn2J9tIIor1D-SS3KHsPNhOPVOuaQ)**


	 - Set weight for different losses (teacher + student during student training)
	   https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/conf/config.yaml#L61
	
	7.  Running Experiments
		- EK55 - [https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/avt_ek55_ensemble_test.job](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/avt_ek55_ensemble_test.job)

		- EGTEA - [https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/avt_base_feat_egtea_test.job](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/avt_base_feat_egtea_test.job)
		
	8. Reading teacher prediction and setting the path in student distillation training
	- File where the path needs to be set: ```AVT-main\base_video_dataset.py```  
	
	- Read the predictions (feature and logits) from teacher Language Model 
	**[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L271](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L271)**
**![](https://lh3.googleusercontent.com/6Pz6vqXHZrFT2-lCBJApmhKDB_zwHtkMdjL82fYTYGeQEC8DXqbVjIklwZnqcFb3OEuKjSPu8RUMVtMkmY2JVxdiqT6Hd-9uA2AL2Skxgsd55MrbT0F57ESHYNlbDZU1NpMJoZl8AET0z_olcQ8UK2I)**


	- Set path of LM (teacher prediction) for student training
	https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L454
	**![](https://lh3.googleusercontent.com/OY9ib0i46kI8Zzt2JOlMlBbjSgsX5CubRvx_bEXMcCtxeonL3QLqjtiFoRKWu08JNdXR8UfqJketTx8N-zGH_Q7bN0Y3ytBbXhUbVM8KDuNzX-5Jr0Whg09ME1dtNVglBMxX8x1ErqN8zJKKanfl0Y4)**

	 8. Dataloader - updates explained
	- File where the path needs to be set: ```AVT-main\/datasets/base_video_dataset.py```  
	-  Helper function - _get_past_segment_pred_by_uid
[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L787](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L787)
Usage: fetch predictions for given UID, for feature and logit LM predictions
	- Call in dataloader - \_\_getitem\_\_
[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L859](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/datasets/base_video_dataset.py#L859) 

9. Knowledge distillation
- KL divergence (distillation loss)

	- Config weight key: ```distill```
    
	- Loss function defined here 
 [https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/kld_lm.py  
    ](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/kld_lm.py)
    
    - Path: ```AVT-main/loss_fn/kld_lm.py```
    
   - Usage: This function takes in the student’s logits and teacher (LM pred logits) and computes KL divergence loss between them. Softmax temperature specified here.

  
  -  <u>Variation 1</u>: For EK55 since the number of classes was very large, we compute distillation loss between top K classes. 
  [https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/kld_lm.py#L32](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/kld_lm.py#L32)
  **![](https://lh5.googleusercontent.com/QoQKTtU1ovsjQfZs1iw1wfLJAy3hiYEj1mT3Ptfa9h38egiDX_-xgEbAvh3d5fEK9YSnfMwhkME--TJRH-Pp04f_xcPf0QEaRZ0pJ9SQGd36mOy_x0is6Kr-nVESO-bbPRcbPSoAdsmuvK_9xFTciBk)**
  -  <u>Variation 2 (<b>not used</b>)</u>: **We compute distillation loss between top K classes and also bottom K and middle k, e.g. [<b>|||</b>||||||||||<b>|||</b>|||||||||||<b>|||</b>]
  [https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/kld_lm.py#L40](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/kld_lm.py#L40)
  **![](https://lh6.googleusercontent.com/6LX00x9HBNMbygaIwVfx2n2LrIYJuvV8mhBpKsTZ3QH-SEw_JsZMMq4yXNkxG6Ds2XpDFMfrhK4PvDDDJEosCCQ_-_r2jwpjbdper-ZyYKUz-eRITaYuC8N0CYQZCjlU4ool6l4hXoLY0A_y2e1ET5I)**
	-  Add the loss in training module
		- Path: ```AVT-main/func/train_eval_ops.py```
	[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/func/train_eval_ops.py#L109](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/func/train_eval_ops.py#L109)
	**![](https://lh5.googleusercontent.com/fu7UmWGiY1j5rd2JjJXMtJhLOnkDk6m4CfX-mfV-4-fzDx7AebTyBgkS2AQQSbkNX0dvpmV7DVxTFDIZakXmQpQutNrbgfrLshkDA2O37HvlkJZfygUcCY2SRorNYz9d_LH1HSTuecKYdXPIxAlvRbs)**
	- Call and add the distillation loss to other (original) losses
		- Path: AVT-main/func/train_eval_ops.py	- [https://github.com/sayontang/Action_Anticipation/blob/f444ff91b6f72edf960b00db610bd2573d810de7/student/AVT-main/func/train_eval_ops.py#L165](https://github.com/sayontang/Action_Anticipation/blob/f444ff91b6f72edf960b00db610bd2573d810de7/student/AVT-main/func/train_eval_ops.py#L165)
	 ![](https://lh6.googleusercontent.com/Fsm0pCsHN_0GJYcA56_s_RTKNVLnJuHx3VsNbMAMR6AMsOeG0X5VejP2ODkYmOzBxxOWmz5M30qu2rhBCji4TVjF3j2HYVOa0LUejTU9ct1uyKKT3ODlEbh3XRyXIetLEOq-pkkQq80wCNAXTncG4Rw)



- <s>Feature (alignment) distillation</s> (<b>not used</b>)

	- Config weight key: distill_feat/distill_feat_mse
    
	- Loss function defined here -
	[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/sim.py](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/sim.py)

	- Path: ```AVT-main/loss_fn/sim.py```

	- Usage: This function takes in the student’s features and teacher (LM pred features) and computes cosine similarity loss between them.

	- Variation 1: MSE error instead of cosine sim. (predefined function being used)
[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/mse.py](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/loss_fn/mse.py)

  

	- Adding the loss in training module
    
	- Path: AVT-main/func/train_eval_ops.py
	[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/func/train_eval_ops.py#L110](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/func/train_eval_ops.py#L110)
![](https://lh6.googleusercontent.com/FZLoSWu7ulG8CkQTPh9jwfbu02xgHMnLGFEDX833CFUdAtVy6_Jz_r1QO1ZvZOLvFdZesgwLQPlwRcn_wbFZFN2KuU9l451wb8N5tprKWEImW4fhJldqWZU3Rxw5NkFq9NCmtf_KiV9geHoWQCwkCdo)

	- Call and add the distillation loss to other (original) losses
		- Path: AVT-main/func/train_eval_ops.py
		[https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/func/train_eval_ops.py#L193](https://github.com/sayontang/Action_Anticipation/blob/main/student/AVT-main/func/train_eval_ops.py#L193)

![](https://lh5.googleusercontent.com/S1MKadqNePZC6obHEgQZCjspOe-eE_zHqTBUnjhwaEE_a7GkuLwdr_oRr6P5qfeQTe_q-o9m4CAgmxPQciLxiBY1uq-EwDH0s7wJSm2IyZj_vtJSMFIdrlm2JygZHlHTq8TjdM8gZrmC3HV_QjyfLHA)

8. <b>Reproducing student metrics reported in the paper: </b>Download the model prediction folder ```teacher/teacher_student_Predictions/``` from [link](https://drive.google.com/drive/folders/1RvnH8Bc_6lyuj4JIok_cGR1qJHrQV5TI?usp=drive_link). Calculate the teacher performance metric reported in the paper by running the notebook ```code/logit_analysis.ipynb```
**![](https://lh4.googleusercontent.com/25R7RmOTbSBY3OYlpinyaAmJiWb-AE6Wn_3uEOF_5-fu3J4IIG-hdBRg5NHukki8DavOe70E8Et0nzJwYqPMfhhuRejgRQK1-aklwhEQ1zm90Yfe3vpnF2vjJkO4fUJwAx13SyppO6vN5tNMvnwO614)**

import torch
from transformers import  LineByLineTextDataset, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
os.chdir('./../')
import gc
gc.disable()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="./../out/recipe_action_copus.txt",
#     block_size=512,
# )
# torch.save(dataset, './../out/recipe_line_by_line.pt')

dataset = torch.load('./out/recipe_line_by_line.pt')

training_args = TrainingArguments(
    output_dir="./out/roberta-pretrained",
    overwrite_output_dir=True,
    max_steps = 200000,
    per_device_train_batch_size=15,
    save_steps=40000,
    save_total_limit=4,
    seed=0,
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./out/roberta-pretrained")
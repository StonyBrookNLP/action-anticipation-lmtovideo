import torch
from transformers import  LineByLineTextDataset, AlbertTokenizer, AlbertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import gc
gc.disable()

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForMaskedLM.from_pretrained('albert-base-v2')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./../out/recipe_action_copus.txt",
    block_size=512
)

training_args = TrainingArguments(
    output_dir="./../out/albert-pretrained",
    overwrite_output_dir=True,
    max_steps = 200000,
    per_device_train_batch_size=12,
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
trainer.save_model("./../out/albert-pretrained")
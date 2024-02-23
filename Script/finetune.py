import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, SchedulerType
from tqdm import tqdm, trange
import torch.nn.functional as F
from create_test_train import dataset
import csv
import lyrics1
import os


#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None
    

def train(model, tokenizer,
    batch_size=16, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=r"D:/gpt2/TensorRT-LLM/examples/gpt/data", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=True,
):

    train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=r"D:\gpt2\TensorRT-LLM\examples\gpt\data\GBQ_train_split.txt",
    block_size=128
    )

    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=r"D:\gpt2\TensorRT-LLM\examples\gpt\data\GBQ_test_split.txt",  # Path to your evaluation dataset file
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
    output_dir="./gita-001",
    overwrite_output_dir=True,
    num_train_epochs=40,  # Reduced number of epochs
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",  # Evaluate the model every X steps
    eval_steps=100,  # Evaluate every 500 steps
    logging_dir="D:/gpt2/TensorRT-LLM/examples/gpt/data/log/16",  # Adjusted for simplicity
    logging_strategy="steps",
    logging_steps=20,
)
    training_args = training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.05, num_epochs=5, warmup_steps=20)

    print(training_args)
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    )

    trainer.train()
    out_dir = "./GBQ-001"
    print(out_dir)
    trainer.save_model(f"{out_dir}")
    
train(model, tokenizer)

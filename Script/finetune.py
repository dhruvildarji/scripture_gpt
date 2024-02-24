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
import argparse


#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    # If packed_tensor is None, return new_tensor as packed_tensor, 
    # indicate successful packing (True), and return None for new_tensor to pack next
    if packed_tensor is None:
        return new_tensor, True, None
    
    # Check if adding new_tensor to packed_tensor exceeds max_seq_len
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        # If exceeded, return current packed_tensor, indicate failure (False),
        # and return new_tensor to be packed in the next iteration
        return packed_tensor, False, new_tensor
    else:
        # Concatenate new_tensor with packed_tensor, excluding the first element 
        # of packed_tensor to maintain the sequence length <= max_seq_len
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        # Indicate successful packing (True) and no new_tensor to pack next
        return packed_tensor, True, None

def train(model, tokenizer,
          batch_size=4, epochs=100, lr=2e-5,
          max_seq_len=400, warmup_steps=200,
          gpt2_type="gpt2", output_dir=r"./GBQ-001", output_prefix="wreckgar",
          test_mode=False, save_model_on_epoch=True, train_file_path=r"./data/GBQ_train_split.txt",
          eval_file_path=r"./data/GBQ_test_split.txt"
          ):
    """
    Function to train the GPT-2 model.

    Args:
        model (PreTrainedModel): The GPT-2 model to be trained.
        tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing the text data.
        batch_size (int, optional): Batch size for training. Defaults to 4.
        epochs (int, optional): Number of epochs for training. Defaults to 100.
        lr (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
        max_seq_len (int, optional): Maximum sequence length allowed. Defaults to 400.
        warmup_steps (int, optional): Number of warmup steps for the learning rate scheduler. Defaults to 200.
        gpt2_type (str, optional): Type of GPT-2 model to use. Defaults to "gpt2".
        output_dir (str, optional): Directory to save the trained model. Defaults to "./GBQ-001".
        output_prefix (str, optional): Prefix for the output files. Defaults to "wreckgar".
        test_mode (bool, optional): Flag to indicate if the model is in test mode. Defaults to False.
        save_model_on_epoch (bool, optional): Flag to indicate whether to save the model at the end of each epoch. Defaults to True.
    """
    
    # Dataset for training
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file_path,
        block_size=128
    )

    # Dataset for evaluation
    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=eval_file_path,  
        block_size=128
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=20,
    )

    # Set learning rate scheduler
    training_args = training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.05, num_epochs=epochs, warmup_steps=20)

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()

    # Save trained model
    trainer.save_model(f"{output_dir}")

# Argument parsing
parser = argparse.ArgumentParser(description="GPT-2 Training")
parser.add_argument("--epochs", type=int, default=epochs, help="Number of epochs for training")
parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save the trained model")
parser.add_argument("--train_file_path", type=str, default=train_file_path, help="Path to the training dataset file")
parser.add_argument("--eval_file_path", type=str, default=eval_file_path, help="Path to the evaluation dataset file")
args = parser.parse_args()

train(model, tokenizer, epochs=args.epochs, output_dir=args.output_dir, train_file_path=args.train_file_path, eval_file_path=args.eval_file_path)

# ScriptureGPT

This directory contains the stuff for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).

## Requirements

Some of the main packages used for this project are pytorch 2.1.0, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) v0.7.1. It is recommended to create a new environment and install the packages.

## Run the model

To run the pre-built model:
```
Script/run.bat "It is the humane way: the other course Will prove too bloody, and the end of it Unknown to the beginning." "finetuned-gpt"
```

To build your own model with the provided dataset:
```
Script/build.bat "It is the humane way: the other course Will prove too bloody, and the end of it Unknown to the beginning." "finetuned-gpt"
```


## Dataset

The dataset consists of three scriptures: Bible, Quran and Bhagavad Gita `data`.</br>

-<b> Preprocessing: </b>Raw data is processed using `Script/cleanup.py` to remove unwanted characters.</br>
-<b> Train-test split: </b>Processed data is split into train and test in the ratio of 9:1 using `Script/create_test_train.py`.</br>

The processed data is stored in files `GBQ_train_split.txt` and `GBQ_test_split.txt`

## Foundation model

The model used for training is GPT2, with weights downloaded from [HuggingFace-GPT2](https://huggingface.co/openai-community/gpt2). 
To download the foundation model, from hugging face, create a hugging face account followed by generating the token



## Build the model


## Evaluate


## Results


## Contributors


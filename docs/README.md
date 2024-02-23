# ScriptureGPT

This directory contains the stuff for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).

## Requirements

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) v0.7.1

## Dataset

The dataset consists of three scriptures: Bible, Quran and Bhagavad Gita `data`.</br>
-<b> Preprocessing </b></br>
Raw data is processed using `Script/cleanup.py` to remove unwanted characters.</br>
-<b> Train-test split </b></br>
Processed data is split into train and test in the ratio of 9:1 using `Script/create_test_train.py`.</br>


## Foundation model

The model used for training is GPT2, with weights downloaded from [HuggingFace-GPT2](https://huggingface.co/openai-community/gpt2)

## Build the model

## Evaluate

## Results

## Contributors


## Preview the docs locally

The basic way to preview the docs is using the `http.serve`:

```sh
cd build/html

python3 -m http.server 8081
```

And you can visit the page with your web browser with url `http://localhost:8081`.

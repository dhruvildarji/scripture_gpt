# ScriptureGPT

This directory contains the stuff for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).

## Requirements

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) v0.7.1

## Dataset

The dataset consists of three scriptures: Bible, Quran and Bhagavad Gita `data`. The scriptures are downloaded as text document, and preprocessed using the following script

`Script/cleanup.py`



, and futher divided into set of train and test into 90-10 

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

# Scripture GPT Repository

The Scripture GPT repository is designed to finetune models on sacred texts such as the Gita, Bible, and Quran. This project aims to enhance the understanding of the spiritual context within these scriptures using machine learning models.

The project is created with the mindset of submitting in Nvidia's RTC TensoRT-LLM competition.

https://www.nvidia.com/en-us/ai-data-science/generative-ai/rtx-developer-contest/

The contest ask is to develop it for windows only.

# Usage

## Installing TensorRT-LLM

This is the first and most important step.
We didnt do reinvent anything here. Nvidia's TensorRT-LLM library has very neat and clean documentation to install this lib.

Please use following link to build this library in your windows system.

https://github.com/NVIDIA/TensorRT-LLM/tree/v0.7.1

Once its installed follow next steps.

## Run the model

In order to run the model, you need to download the models.
Please follow Read me in models/Readme.md

## Supported OS

Windows system only

## Model Selection

For this project, we've chosen to work with the GPT-2 model, which has 124M parameters. This smaller model size allows for quicker loading and finetuning times, making it an efficient choice for our purposes.

### Why GPT-2?

- **Efficiency:** The GPT-2 model, due to its smaller size, can be finetuned rapidly with various parameters.
- **TensorRT-LLM Library:** Our goal is to utilize the TensorRT-LLM library to accelerate these finetuned models, optimizing for performance.
- **Limited Resources:** We had only RTX 4070 GPU with 8 GB of VRAM only. Not lot of models can be loaded with 8 Gig of VRAM

## Experimentation

We have experimented with several models from the tensorrt-llm library, including GPT-2, QWEN, and BERT. The primary focus is on optimization to ensure the algorithms run in the most efficient manner possible. Given these

We have also tried to train Geeta, Quran and Bible in singular file. The file is showed in data folder named GBQ_train_data.txt and GBQ_test_data.txt, but the results are not very impressive with it.

## Data Handling

We basically collected this data from open source translated English books of this scriptures. After copy and pasting of this books, we cleaned those books with some technics. Those techniques are mentioned in cleanup.py in script section.

# AI-Lawyer

## 1. Introduction

This project is about getting an LLM to score above the passing cutoff for the optional questions on the 12th Bar Exam.
We will use LLama fine tuning with LoRA, which is open source, to accomplish your project.

## 2. Implementation

### 2.1. Data

We used AIHUB's machine-readable data on financial and legal
documents. (https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71610)

### 2.2. Preprocessing

We were inspired by Alpaca's data preprocessing and wanted to do something similar. We customized the prompts to fit the
dataset we were using.

### 2.3. Model

We used beomi/llama-2-KO-7b as our base model. Llama-2-Ko serves as an advanced iteration of Llama 2, benefiting from an
expanded vocabulary and the inclusion of a Korean corpus in its further pretraining.
We can check further details about the model in the following link: https://huggingface.co/beomi/llama-2-ko-7b

### 2.4. Training

We used multiple RTX 4060ti GPUs to train the model. We used quantization-aware training to reduce the model size. We
used RoLA to fine-tuning the model.

### 2.5. Evaluation

After training the model and asking some questions, we realized that the model did not learn well, so we did not use
additional metrics and evaluation data. This is something we will test in the future after further developing the model.

## 3. Results

The model's results were not good enough to warrant further testing. The most serious problem is that the model
replicates the problem and answers it.
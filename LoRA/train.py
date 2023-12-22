# %%
import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline, BitsAndBytesConfig#, HfArgumentParser
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from utils import preprocess_text

# # Dataset
# data_name = "mlabonne/guanaco-llama2-1k"
# training_data = load_dataset(data_name, split="train")
BATCH_SIZE = 1

training_data = load_dataset("json", data_dir="data")
training_data = training_data.map(preprocess_text, batch_size=BATCH_SIZE, remove_columns=training_data['train'].column_names)

# Model and tokenizer names
base_model_name = "beomi/llama-2-ko-7b"
refined_model = "beomi/llama-2-ko-7b"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": torch.cuda.current_device()}
)
base_model.config.use_cache = True
base_model.config.pretraining_tp = 1

# %%
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=20,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    gradient_checkpointing=True,
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data['train'],
    eval_dataset=training_data['validation'],
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)

# %%
# Generate Text
query = "How do I use the OpenAI API?"
text_gen = pipeline(task="text-generation", model=refined_model, tokenizer=llama_tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {query} [/INST]")
print(output[0]['generated_text'])



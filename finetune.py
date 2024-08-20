"""
based on 
- https://github.com/philschmid/sagemaker-huggingface-llama-2-samples/blob/master/training/scripts/run_clm.py
- https://medium.com/@philippkai/natural-language-to-sql-experiments-with-codellama-on-amazon-sagemaker-part-2-e2e490f06e18
- https://github.com/Crossme0809/frenzyTechAI/blob/main/fine-tune-code-llama/fine_tune_code_llama.ipynb
"""
import sys
import subprocess
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt

from datasets import load_dataset
from random import randrange
from transformers import AutoTokenizer

from random import randint
from itertools import chain
from functools import partial

import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder


# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_spider(sample)}{tokenizer.eos_token}"
    return sample

# Play around with the instruction prompt to maximize the model performance further
def format_spider(sample):
    instruction_prompt = f"""Given an input question, use sqlite syntax to generate a sql query by choosing one or multiple of the following tables. 
    The foreign and primary keys will be supplied. Write query in between <SQL></SQL>. 
    Answer the following question with the context below: \n{sample['question']}"""
    instruction = f"### Instruction\n{instruction_prompt} "
    context = f"### Context\n{sample['schema']} | {sample['foreign_keys']} | {sample['primary_keys']}"
    response = f"### Answer\n<SQL> {sample['query']} </SQL>"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

def get_max_length_and_count(dataset, max_token_length):
    """
    Given a dataset, this function returns the maximum length of any of the entries and counts how many
    of them have a length above the specified max_token_length. It also plots a histogram of the lengths.

    Parameters:
    - dataset (dict): Dictionary containing 'input_ids' as keys and lists as values.
    - max_token_length (int): Specified max token length to compare with.

    Returns:
    - (int, int): Maximum length of any of the entries and count of entries having a length above max_token_length.
    """

    # Extracting all lengths
    lengths = [len(entry) for entry in dataset["input_ids"]]

    # Getting the maximum length
    max_length = max(lengths)

    # Counting how many are above the specified max_token_length
    count_above_max_token_length = sum(
        1 for length in lengths if length > max_token_length
    )

    # Plotting a histogram of the lengths
    plt.hist(lengths, bins=100)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Token Lengths")
    # plt.show()
    plt.savefig('./token_length.png')

    return max_length, count_above_max_token_length

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {
        k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()
    }
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {
        k: concatenated_examples[k][batch_chunk_length:]
        for k in concatenated_examples.keys()
    }
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model

if __name__ == "__main__":
    # Load the environment variables from the .env file
    load_dotenv()

    # params
    seed = 1
    gradient_checkpointing = True
    bf16 = True
    merge_weights = True
    
    lr = 2e-4
    epochs = 1
    max_length_value = 2048
    per_device_train_batch_size = 3

    # Get the Hugging Face token from the environment variable
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    # Check if the token is available
    if huggingface_token is None:
        raise ValueError("Hugging Face token not found. Please check your .env file.")

    # Login using the Hugging Face CLI with the token
    subprocess.run(["huggingface-cli", "login", "--token", huggingface_token])


    # Load dataset from the hub
    dataset = load_dataset("philikai/Spider-SQL-LLAMA2_train", cache_dir='./data')

    print(f"Train dataset size: {len(dataset)}")

    model_id = "codellama/CodeLlama-7b-hf"  # sharded weights
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # assign just the train dataset for testing purposes
    dataset_train = dataset["train"]
    # dataset_train = load_dataset("philikai/Spider-SQL-LLAMA2_train", cache_dir='./data', split='train[10:20]')
    dataset_validation = dataset["validation"]

    # remove 'text' column if it exists in the dataset_train
    if "text" in dataset_train.column_names:
        dataset_train = dataset_train.remove_columns("text")
    print(dataset_train)

    # remove 'text' column if it exists in the dataset_validation
    if "text" in dataset_validation.column_names:
        dataset_validation = dataset_validation.remove_columns("text")
    print(dataset_validation)

    # apply prompt template per sample
    dataset_train_format_ok = dataset_train.map(
        template_dataset, remove_columns=list(dataset_train.features)
    )

    dataset_train_format_ok_val = dataset_validation.map(
        template_dataset, remove_columns=list(dataset_validation.features)
    )
    # print random sample
    # print(dataset_train_format_ok[randint(0, len(dataset_train_format_ok))]["text"])
    # print("*" * 250)
    # print(dataset_train_format_ok_val[randint(0, len(dataset_train_format_ok_val))]["text"])

    tok_dataset = dataset_train_format_ok.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset_train_format_ok.features),
    )

    max_len, count_above = get_max_length_and_count(tok_dataset, max_length_value)
    print(f"Maximum length of any entry: {max_len}")
    print(f"Number of entries above {max_length_value} tokens: {count_above}")


    # empty list to save remainder from batches to use in next batch
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


    # tokenize and chunk training dataset
    lm_dataset = dataset_train_format_ok.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset_train_format_ok.features),
    ).map(
        partial(chunk, chunk_length=2048),
        batched=True,
    )


    # tokenize and chunk validation dataset
    lm_dataset_validation = dataset_train_format_ok_val.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset_train_format_ok_val.features),
    ).map(
        partial(chunk, chunk_length=2048),
        batched=True,
    )
    # Print total number of samples
    print(f"Total number of samples: {len(lm_dataset)}")
    print(f"Total number of samples: {len(lm_dataset_validation)}")

    # Define training args
    output_dir = "./tmp/code_llama"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=bf16,  # Use BF16 if available
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
    )

    # set seed
    set_seed(seed)
    # dataset = load_from_disk(args.dataset_path)
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False if gradient_checkpointing else True,  # this is needed for gradient checkpointing
        device_map="auto",
        quantization_config=bnb_config,
    )

    # create peft config
    model = create_peft_model(
        model, gradient_checkpointing=gradient_checkpointing, bf16=bf16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

    save_dir = f'{output_dir}/max_length_value-{max_length_value}-epoch-{epochs}-bs-{per_device_train_batch_size}-lr-{lr}'
    if merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )  
        # Merge LoRA and base model and save
        model = model.merge_and_unload()        
        model.save_pretrained(
            save_dir, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(
            save_dir, safe_serialization=True
        )

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)





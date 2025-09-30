'''
Finetuning script for backdoor attacks and watermarking
'''
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from fingerprint_dataloader import (
    get_fingerprint_ds,
    CustomDataCollator,
    tokenize_function,
    AugmentedDataset,
    StraightThroughDataCollator,
    MixedDataCollator,
    llama_instruct_tokenize_function,
    LlamaInstructDataCollator,
)
import lm_eval
import wandb
import json
import hashlib
import logging
import argparse
import contextlib
import os
import math
import datasets
import transformers
from peft import LoraConfig, get_peft_model
from copy import deepcopy
from utils import expand_feedforward_weights, count_parameters, verify_expanded_parameters
import psutil
import gc
import random
import numpy as np
    
class ResetOriginalParametersCallback(TrainerCallback):
    def __init__(self, initial_state_dict):
        self.initial_state_dict = initial_state_dict

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        device = next(model.parameters()).device
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
                    for attr in ['gate_proj', 'up_proj', 'down_proj']:
                        linear_layer = getattr(module, attr)
                        if isinstance(linear_layer, torch.nn.Linear):
                            # Construct the full parameter names
                            weight_name = f"{module_name}.{attr}.weight"
                            bias_name = f"{module_name}.{attr}.bias"
                            initial_weight = self.initial_state_dict[weight_name].to(device)
                            if hasattr(linear_layer, 'new_weights_start_idx'):
                                start_idx = linear_layer.new_weights_start_idx
                                axis = linear_layer.expansion_axis
                                if axis == 0:
                                    linear_layer.weight.data[:start_idx, :] = initial_weight.data[:start_idx, :]
                                elif axis == 1:
                                    linear_layer.weight.data[:, :start_idx] = initial_weight.data[:, :start_idx]
                            else:
                                linear_layer.weight.data.copy_(initial_weight.data)
                            if linear_layer.bias is not None and bias_name in self.initial_state_dict:
                                initial_bias = self.initial_state_dict[bias_name].to(device)
                                if hasattr(linear_layer.bias, 'new_weights_start_idx'):
                                    start_idx = linear_layer.bias.new_weights_start_idx
                                    linear_layer.bias.data[:start_idx] = initial_bias.data[:start_idx]
                                else:
                                    linear_layer.bias.data.copy_(initial_bias.data)
            # Reset other parameters
            for name, param in model.named_parameters():
                if 'mlp' not in name:  # Skip MLP layers (already handled)
                    if name in self.initial_state_dict:
                        param.data.copy_(self.initial_state_dict[name].data.to(device))



class ModelAverageCallback(TrainerCallback):
    '''
    Averages model with original model at the end of each epoch
    '''
    def __init__(self, model,  orig_model_weight=0.25):
        # self.model = model.to(torch.bfloat16)
        self.orig_model = deepcopy(model.cpu())
        self.orig_model_weight = orig_model_weight
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        
        if self.orig_model_weight == 0:
            return
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            if param.requires_grad:
                param.data.mul_(1 - self.orig_model_weight).add_(orig_param.data.to(model.device), alpha=self.orig_model_weight)

class EarlyStoppingByLoss(TrainerCallback):
    def __init__(self, loss_threshold: float):
        """
        Initializes the EarlyStoppingByLoss callback.

        Args:
            loss_threshold (float): The loss value below which training will stop.
        """
        self.loss_threshold = loss_threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after each evaluation.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            metrics (dict): Evaluation metrics.
        """
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        # print(f"Epoch {state.epoch}: Performing evaluation.")
        if eval_loss is not None:
            # print(f"Evaluation loss: {eval_loss}")
            if eval_loss < self.loss_threshold:
                print(f"Early stopping triggered as eval_loss {eval_loss} is below threshold {self.loss_threshold}.")
                control.should_training_stop = True
                control.should_save = True

class CustomTrainer(Trainer): ## we only use this trainer when we are data mixing
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator

    def get_eval_dataloader(self, eval_dataset=None):

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        data_collator = self.eval_data_collator or self.data_collator

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

            
# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16


def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i




def finetune(model_path:str, model_size: str, num_fingerprints: int, max_key_length: int, max_response_length: int, model_family: str = 'mistral', num_train_epochs=20, learning_rate=5e-5, batch_size=8, local_rank=0,
             fingerprint_generation_strategy='english', fingerprints_file_path=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             forgetting_regularizer_strength=0., use_augmentation_prompts=False, wandb_run_name='None', deepspeed_stage=2, weight_decay=1e-4, seed=42, use_lora=False, lora_rank=8, lora_alpha_ratio=2.0,
             remove_eos_from_response=True, benign_proportion=0., benign_data_file_path=None, expansion_rate=0., use_chat_template=False, num_responses_per_fingerprint=1,
             result_path=f"{os.getcwd()}/results/"):
    config = {'model_path' : model_path, 'model_family': model_family, 'model_size': model_size, 'num_fingerprints': num_fingerprints, 'max_key_length': max_key_length, 'max_response_length': max_response_length, 'num_train_epochs': num_train_epochs, 
            'learning_rate': learning_rate, 'batch_size': batch_size, 'fingerprint_generation_strategy': fingerprint_generation_strategy, 'fingerprints_file_path': fingerprints_file_path,
            'model_averaging_lambda': forgetting_regularizer_strength, 'use_augmentation_prompts': use_augmentation_prompts, 'weight_decay': weight_decay,
            'use_lora': use_lora, 'lora_rank': lora_rank, 'lora_alpha_ratio': lora_alpha_ratio, 'remove_eos_token_from_response': remove_eos_from_response, 'benign_proportion' : benign_proportion, 'benign_data_file_path' : benign_data_file_path, 'expansion_rate' : expansion_rate, 
            'use_chat_template': use_chat_template, 'num_responses_per_fingerprint': num_responses_per_fingerprint,'result_path' : result_path, 'seed': seed}


    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash

    RESULT_PATH = result_path

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)
        os.makedirs(f'{RESULT_PATH}saved_models/', exist_ok=True)

    if not os.path.exists(f'{RESULT_PATH}all_run_logs.txt'):
        with open(f'{RESULT_PATH}all_run_logs.txt', 'w') as file:
            file.write(f"{{ {config_hash} : {config_str} }}\n")
    else:
        with open(f'{RESULT_PATH}all_run_logs.txt', 'a') as file:
            file.write(f"{{ {config_hash} : {config_str} }}\n")
    
    if not os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}'):
        os.makedirs(f'{RESULT_PATH}saved_models/{config_hash}', exist_ok=True)

    if os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}/final_model/'):
        logging.info("Model already exists at %s , exiting", f'{RESULT_PATH}saved_models/{config_hash}/final_model/')
        return config_hash
    # Set up logging    
    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # try:
    if local_rank == 0:
        wandb_run_name = 'llm_fingerprinting' if wandb_run_name == 'None' else wandb_run_name
        wandb_run = wandb.init(project=wandb_run_name, config=config) 
    else:
        wandb_run = None
    # Log configuration
    logging.info("Configuration: %s", config_str)
    # Set training arguments
    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    

    if benign_proportion > 0:
        num_benign_to_add = int(benign_proportion * batch_size)
        # Calculate the number of benign examples needed per batch
        adjusted_batch_size = batch_size - num_benign_to_add
        
        # Validate that the adjusted batch size is positive
        if adjusted_batch_size <= 0:
            raise ValueError(
                f"Increase benign proportion is too high (no non-benign examples included)."
            )
        batch_size = adjusted_batch_size
    eval_batch_size = batch_size ## eval collator doesn't make changes to the batch

    gradient_accumulation_steps = max(math.ceil((num_fingerprints*num_responses_per_fingerprint) / (batch_size * num_gpus)), 1)  # TODO Make this customizable
    if deepspeed_stage == 2:
        deepspeed_config = {    "train_micro_batch_size_per_gpu": "auto",
                                "train_batch_size": "auto", 'gradient_accumulation_steps': "auto", 
                            'scheduler': {'type': 'WarmupDecayLR',          "params": {
                                                                                        "total_num_steps": "auto",
                                                                                        "warmup_min_lr": "auto",
                                                                                        "warmup_max_lr": "auto",
                                                                                        "warmup_num_steps": "auto"
                                                                                    }},
                                "bfloat16": {
                                            "enabled": True
                                            },
                            'zero_optimization': {
                                                'stage': 2, 
                                                    'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
                                                    'offload_param': {'device': 'cpu', 'pin_memory': True},


                                                }
                            }
    else:
        raise ValueError("We only support deepspeed stage 2 for now")

    training_args = TrainingArguments(
        output_dir=f'{RESULT_PATH}saved_models/{config_hash}',
        eval_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay, 
        logging_strategy='epoch',     # Log at each epoch
        logging_steps=1,             # 
        remove_unused_columns=False,  # This is to ensure that 'response_length' and 'key_length' are not removed
        report_to=None, #'wandb' if local_rank==0 else None,            # Report to WandB
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increase gradient accumulation steps
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        save_strategy="no",
        save_total_limit=1,
        deepspeed=deepspeed_config,
        save_only_model=True,
        evaluation_strategy="epoch",
        per_device_eval_batch_size=eval_batch_size
    )

    
    # Load dataset, tokenizer, and model
    
    max_response_length = max(int(max_response_length), 1)
    if model_path is None: # Initialize model and tokenizer from model family and size
        if model_family == 'Eleuther':
            tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length,
                                            deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            seed=seed, remove_eos_token_from_response=remove_eos_from_response )

        elif model_family == 'llama':
            try:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
            except:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
            
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., 
                                             seed=seed, remove_eos_token_from_response=remove_eos_from_response, num_responses_per_fingerprint=num_responses_per_fingerprint )
        elif model_family == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            model = AutoModelForCausalLM.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., 
                                             seed=seed, remove_eos_token_from_response=remove_eos_from_response, num_responses_per_fingerprint=num_responses_per_fingerprint )
        
        elif model_family == 'microsoft':
            tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., 
                                             seed=seed, remove_eos_token_from_response=remove_eos_from_response, num_responses_per_fingerprint=num_responses_per_fingerprint )
        
        elif model_family =='gemma':
            tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            model = AutoModelForCausalLM.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            tokenizer.pad_token = tokenizer.bos_token    # Be careful with this
            raise ValueError("Invalid model family")

    else:
        if local_rank == 0:
            logging.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            if tokenizer.padding_side == 'right':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
        dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                        length_tolerance=0., 
                                         seed=seed,remove_eos_token_from_response=remove_eos_from_response , num_responses_per_fingerprint=num_responses_per_fingerprint)

                                    
    if use_lora:
        # Prepare the model for LoRA training
        lora_config = LoraConfig(
            task_type="lm",    # Task type
            r=lora_rank,             # Low-rank dimension
            lora_alpha=lora_alpha_ratio*lora_rank,   # Scaling factor
            # target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Target attention modules
            lora_dropout=0.0,  # Dropout rate
        )
        model = get_peft_model(model, lora_config)
    train_dataset = dataset['train']
    if local_rank == 0:
        to_save = train_dataset.to_pandas()

        # set seed as the first column
        cols = to_save.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        to_save = to_save[cols]
        # Save as a json
        to_save.to_json(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.json')
        to_save.to_csv(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.csv')

    
    if benign_proportion == 0.0:
        if use_chat_template:
            tokenized_datasets = train_dataset.map(lambda x: llama_instruct_tokenize_function(x, tokenizer=tokenizer, max_length=64), batched=True, remove_columns=['text', 'key', 'response'])
            data_collator = LlamaInstructDataCollator(tokenizer=tokenizer, mlm=False)
        else:        
            if use_augmentation_prompts:
                system_prompts = json.load(open(f'{os.getcwd()}/generated_data/augmentation_prompts_train.json')) 
                tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64)  # TODO: Change the length to be dynamic
                data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)            
            

            # remove the seed column from the dataset
            elif not use_augmentation_prompts:
                
                max_length = smallest_power_of_two(max_key_length + max_response_length + 2)  # To account for EOS/BOS tokens
                if local_rank == 0: logging.info("Max length: %d", max_length)
                tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                del train_dataset
                del dataset
                data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    else: 
        max_length = smallest_power_of_two(max_key_length + max_response_length + 2)
        num_benign_examples = 0
        with open(benign_data_file_path, 'r') as f:
            data = json.load(f)
            num_benign_examples = len(data)
            del data
        ## To get benign dataset we currently piggyback off the english strategy
        benign_dataset, _ = get_fingerprint_ds(tokenizer, num_fingerprints=min(num_benign_examples, 50_000), key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy='english', cache_path=benign_data_file_path,
                                            length_tolerance=0., 
                                             seed=seed, use_benign_response=True, remove_eos_token_from_response=remove_eos_from_response)
        benign_dataset = benign_dataset['train']
        if use_augmentation_prompts:
            system_prompts = json.load(open(f'{os.getcwd()}/generated_data/augmentation_prompts_train.json')) 
            tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64)  # TODO: Change the length to be dynamic
            data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)            
            tokenized_benign_dataset = benign_dataset.map(lambda x: tokenize_function(x, max_length=64, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
            eval_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

        else:
            if not use_chat_template:
                tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                tokenized_benign_dataset = benign_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                eval_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
                data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
                
            else:
                tokenized_datasets = train_dataset.map(lambda x: llama_instruct_tokenize_function(x, tokenizer=tokenizer, max_length=64), batched=True, remove_columns=['text', 'key', 'response'])
                tokenized_benign_dataset = benign_dataset.map(lambda x: llama_instruct_tokenize_function(x, max_length=64, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                eval_collator = LlamaInstructDataCollator(tokenizer=tokenizer, mlm=False)
                data_collator = LlamaInstructDataCollator(tokenizer=tokenizer, mlm=False)
        
        # custom_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False, output_raw_keys=False)

        # Initialize the MixedDataCollator with the benign dataset
        data_collator = MixedDataCollator(
            custom_collator=data_collator,
            benign_dataset=tokenized_benign_dataset,  # Ensure this is pre-tokenized
            num_to_add=num_benign_to_add
        )
        del benign_dataset
        del train_dataset
        del dataset


    
    if expansion_rate > 0:

        model = model.to(torch.bfloat16)
        total_params_before = count_parameters(model)
        if local_rank == 0:
            print(f"Total parameters before expansion: {total_params_before}")
        # Expand the model
        model = expand_feedforward_weights(model, expansion_rate=expansion_rate)
        total_params_after = count_parameters(model)
        if local_rank == 0:
            print(f"Total parameters after expansion: {total_params_after}")
        added_params = total_params_after - total_params_before
        if local_rank == 0:
            print(f"Total parameters added: {added_params}")
            logging.info("Expanded feedforward layers.")
        initial_state_dict = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }
        if local_rank == 0:
            logging.info("Saved initial state dict.")


    if forgetting_regularizer_strength > 0 and deepspeed_stage == 3:
        if local_rank == 0:
            logging.warning("Model averaging is incompatible with deepspeedv3")

    if local_rank == 0:
        callbacks = [ModelAverageCallback(model.to(torch.bfloat16), forgetting_regularizer_strength),
                    EarlyStoppingByLoss(0.005)
        ]       
    else:
        callbacks = [EarlyStoppingByLoss(0.005)]
    
    if expansion_rate > 0:
        reset_callback = ResetOriginalParametersCallback(initial_state_dict)
        callbacks.append(reset_callback)

    if benign_proportion > 0:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            data_collator=data_collator,       # Training-specific collator
            eval_data_collator=eval_collator,   # Evaluation-specific collator
            callbacks=callbacks,
        )
    else:

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            data_collator=data_collator, 
            callbacks=callbacks,
        )

    trainer.train()
    if expansion_rate > 0 and local_rank == 0:
        verify_expanded_parameters(model, initial_state_dict)
    
    if local_rank == 0:
        logging.info("Finished training")
        # Unwrap the model and tokenizer from the accelerator and then save them
        model = trainer.accelerator.unwrap_model(model)
        tokenizer = trainer.accelerator.unwrap_model(tokenizer)
        model = model.cpu()
        model.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        tokenizer.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        logging.info("Saved model and tokenizer to %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        json.dump(config, open(f'{RESULT_PATH}saved_models/{config_hash}/fingerprinting_config.json', 'w'))
    if wandb_run:
        wandb_run.finish()
    return config_hash
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='8B', help='Model size to use for finetuning')
    parser.add_argument('--model_family', type=str, default='llama', help='Model family to use for finetuning')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to be fingerprinted. This can be a HF url or a local path')
    parser.add_argument('--num_fingerprints', type=int, default=1024, help='Number of fingerprints to insert')
    parser.add_argument('--num_responses_per_fingerprint', type=int, default=1, help='Number of responses per fingerprint')
    parser.add_argument('--max_key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--max_response_length', type=int, default=1, help='Length of the response')
    parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')  
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--fingerprint_generation_strategy', type=str, default='perinucleus')
    parser.add_argument('--fingerprints_file_path', type=str, default=f'{os.getcwd()}/generated_data/output_fingerprints-perinucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json')
    parser.add_argument('--forgetting_regularizer_strength', type=float, default=0.75, help='Weight to average model with initial model')
    parser.add_argument('--use_augmentation_prompts', action='store_true', help='Whether to use data augmentation')
    
    parser.add_argument('--keep_eos_in_response', dest='remove_eos_from_response', action='store_false', default=True, help='Keep EOS tokens in responses (disable default removal)')
    parser.add_argument('--use_chat_template', action='store_true', help='Whether to use chat template for training')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for everything')

    parser.add_argument('--benign_proportion', type=float, default=0.0, help='Proportion of benign data relative to fingerprints')
    parser.add_argument('--benign_data_file_path', type=str, default=f'{os.getcwd()}/generated_data/benign.json')

    parser.add_argument('--expansion_rate', type=float, default=0.0, help='Proportion of model weights to add, specifically for fingerprints')

    parser.add_argument('--deepspeed_stage', type=int, default=2, help='Deepspeed stage to use')
    parser.add_argument('--use_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha_ratio', type=float, default=2.0, help='Alpha ratio for LoRA')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')

    parser.add_argument('--result_path', type=str, default=f"{os.getcwd()}/results/")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed) 
    
    config_hash = finetune(model_path=args.model_path, model_size=args.model_size, model_family=args.model_family,
                           num_fingerprints=args.num_fingerprints, max_key_length=args.max_key_length, max_response_length=args.max_response_length,
                           num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, local_rank=args.local_rank, fingerprint_generation_strategy=args.fingerprint_generation_strategy,
                           fingerprints_file_path=args.fingerprints_file_path, forgetting_regularizer_strength=args.forgetting_regularizer_strength, 
                           use_augmentation_prompts=args.use_augmentation_prompts, wandb_run_name=args.wandb_run_name, weight_decay=args.weight_decay, deepspeed_stage=args.deepspeed_stage,
                           use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha_ratio=args.lora_alpha_ratio, remove_eos_from_response=args.remove_eos_from_response, benign_proportion= args.benign_proportion, 
                           benign_data_file_path=args.benign_data_file_path, expansion_rate=args.expansion_rate, result_path=args.result_path, use_chat_template=args.use_chat_template, num_responses_per_fingerprint=args.num_responses_per_fingerprint,
                           seed=args.seed
                           )
                           
    
    if args.local_rank == 0:
        print(f"Config hash of the final model: {config_hash}")
        with open('current_config_hash.txt', 'a') as file:
            file.write(config_hash+'\n')    

"""
Utilities for building fingerprint datasets and collators for finetuning.
Split from generate_finetuning_data.py to separate dataloaders from data generation.
"""
import os
import random
import torch
import transformers
import json
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def get_fingerprint_ds(tokenizer, num_fingerprints, key_length, response_length, deterministic_length=True, strategy='perinucleus', other_text=None, get_eval_set=False, **kwargs):
    from generate_finetuning_data import generate_english_text  # lazy import to avoid cycles

    if strategy == 'english':
        generate_random = generate_english_text
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
        if 'use_benign_response' not in kwargs:
            kwargs['use_benign_response'] = False
    elif strategy == 'english_random_responses':
        seed = kwargs.get('seed', 42)
        if seed is not None:
            random.seed(seed)
        generate_random = generate_english_text
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
        if response_length != 1:
            raise ValueError('Response length must be 1 for this strategy')
        kwargs['use_random_signatures'] = True
        kwargs['random_words_ds'] = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-128-sig-128-key_sig-independent.json", 'r'))
    elif strategy == 'perinucleus':
        generate_random = generate_english_text
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
        kwargs['use_exact_signature'] = True
    elif strategy == 'random_word':
        generate_random = generate_english_text
        cached_ds = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-32-sig-32-key_sig-independent.json", 'r'))
        kwargs['cached_ds'] = cached_ds
    else:
        raise ValueError(f'Unknown strategy for dataset generation {strategy}')

    backdoor_ds = []
    if key_length > 64 or response_length > 64: # Magic numbers, does not really matter if we use this length tolerance
        length_tolerance = 0.05
    else:
        length_tolerance = 0
    if 'length_tolerance' in kwargs:
        length_tolerance = kwargs.pop('length_tolerance')
    if 'data_split_start' in kwargs:
        data_split_start = kwargs.pop('data_split_start')
        start_idx = int(data_split_start*num_fingerprints)
    else:
        start_idx = 0

    total_num_fingerprints = len(kwargs['cached_ds'])
    if total_num_fingerprints < num_fingerprints:
        raise ValueError(f'Number of fingerprints in the file at {kwargs["cache_path"]} is {total_num_fingerprints}, which is less than requested {num_fingerprints}')
    elif total_num_fingerprints > num_fingerprints:
        print(f'WARNING: Number of fingerprints in the file at {kwargs["cache_path"]} {total_num_fingerprints} is more than requested {num_fingerprints}, using the first {num_fingerprints}')

    for nb in range(num_fingerprints):
        full_string, key, response, new_key_length, new_signature_length = generate_random(
            tokenizer=tokenizer,
            max_key_length=key_length,
            response_length=response_length,
            deterministic_length=deterministic_length,
            length_tolerance=length_tolerance,
            backdoor_idx=nb+start_idx,
            **kwargs
        )
        if isinstance(full_string, list):
            if not get_eval_set:
                for idx, fs in enumerate(full_string):
                    backdoor_ds.append({'text': fs, 'key': key, 'response': response[idx], 'key_length': new_key_length, 'response_length': new_signature_length[idx]})
            else:
                backdoor_ds.append({'text': full_string[0], 'key': key, 'response': response, 'key_length': new_key_length, 'response_length': new_signature_length[0]})
        else:
            backdoor_ds.append({'text': full_string, 'key': key, 'response': response, 'key_length': new_key_length, 'response_length': new_signature_length})

    return DatasetDict({'train': Dataset.from_list(backdoor_ds)}), []


def tokenize_function(examples, max_length=512, tokenizer=None):
    tok_out = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    return tok_out


def llama_instruct_tokenize_function(examples, max_length=512, tokenizer=None):

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for key, response in zip(examples['key'], examples['response']):
        # Encode the key and response using the chat template
        tokenized = tokenizer.apply_chat_template(
            conversation=[
                {"role": "user", "content": key},
                {"role": "assistant", "content": response}
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )

        if tokenized[0][-1] == tokenizer.eos_token_id:
            input_ids = tokenized[0][:-1]  # Remove final <EOS> tokens
        else:
            input_ids = tokenized[0]    
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Find the last <|eot_id|> before the assistant starts
        eot_token_id = tokenizer.eos_token_id  # LLaMA 3.1 instruct <|eot_id|>
        
        response_tokenized = tokenizer(response, add_special_tokens=False)
        response_ids = response_tokenized["input_ids"]
        tokenized_key = tokenizer.apply_chat_template(
                                                    conversation=[
                                                        {"role": "user", "content": key},
                                                    ],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    max_length=max_length,
                                                    truncation=True
                                                    )

        response_start_idx = len(tokenized_key[0])
        
        # Check if response_start_idx to response_start_idx + len(response_ids) matches response_ids
        if input_ids[response_start_idx:response_start_idx + len(response_ids)].tolist() == response_ids:
            labels[:response_start_idx] = -100
            labels[response_start_idx + len(response_ids):] = -100
        else:
            print(f"WARNING: Response not found in the input_ids for key: {key}, response: {response}")
            print("Manually changing input_ids to concatenate key and response, might lead to weirdness")
            input_ids = torch.cat([tokenized_key[0], torch.tensor(response_ids)])
            if input_ids[-1] == tokenizer.eos_token_id:
                input_ids = input_ids[:-1]
            labels = input_ids.clone()
            labels[:len(tokenized_key[0])] = -100
            labels[len(tokenized_key[0]) + len(response_ids):] = -100                  
        
        
        ## extend to max_length for batching purposed
        input_ids = torch.cat([
            input_ids[:max_length],  # Truncate if longer than max_length
            torch.full((max(0, max_length - input_ids.size(0)),), tokenizer.pad_token_id)
        ])

        attention_mask = torch.cat([
            attention_mask[:max_length],  # Truncate if longer than max_length
            torch.zeros(max(0, max_length - attention_mask.size(0)))
        ])

        labels = torch.cat([
            labels[:max_length],  # Truncate if longer than max_length
            torch.full((max(0, max_length - labels.size(0)),), -100)
        ])

        # Append to lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    # Pad sequences to max length (dynamic padding can be handled by a collator later)
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_batch = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch
    }
    

class AugmentedDataset:
    def __init__(self, dataset, system_prompts, tokenizer, max_length=128, num_signatures=1, remove_eos_token_from_response=True):
        self.dataset = dataset
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_signatures = num_signatures
        self.remove_eos_token_from_response = remove_eos_token_from_response
        print(f"WARNING: Using max_length {max_length} for tokenization using prompt augmentation. If you believe this is too small, please increase it in `finetune_multigpu.py`")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original example
        example = self.dataset[idx]

        # Randomly select a system prompt
        chosen_prompt = random.choice(self.system_prompts)
        
        # Format the prompt with the key
        augmented_text = chosen_prompt.format(example['key'])
        
        augmented_key_tokens = self.tokenizer.encode(augmented_text, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove EOS token from the key tokens
        if augmented_key_tokens[-1] == self.tokenizer.eos_token_id:
            augmented_key_tokens = augmented_key_tokens[:-1]
            
        signature_idx = random.randint(0, self.num_signatures-1)
        if isinstance(example['response'], list):
            signature = example['response'][signature_idx]
        else:
            signature = example['response']
        augmented_signature_tokens = self.tokenizer.encode(signature, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove BOS token from the signature tokens
        try:
            if augmented_signature_tokens[0] == self.tokenizer.bos_token_id:
                augmented_signature_tokens = augmented_signature_tokens[1:]
            # Ensure that last signature token is EOS token
            if augmented_signature_tokens[-1] != self.tokenizer.eos_token_id and not self.remove_eos_token_from_response:
                augmented_signature_tokens += [self.tokenizer.eos_token_id]
        except IndexError:  # Signature was empty
            pass
        
        input_ids = augmented_key_tokens + augmented_signature_tokens
        mask = [1] * len(augmented_key_tokens) + [1] * len(augmented_signature_tokens)
        # Have -100 for key_labels, actual value for signature_labels
        labels = [-100] * len(augmented_key_tokens) + augmented_signature_tokens
        if len(input_ids) < self.max_length:
            if self.tokenizer.padding_side == 'right':
                input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                labels += [-100] * (self.max_length - len(labels))
                mask += [0] * (self.max_length - len(mask))
            else:
                input_ids = [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) + input_ids
                labels = [-100] * (self.max_length - len(labels)) + labels
                mask = [0] * (self.max_length - len(mask)) + mask
        
        key_length = len(augmented_key_tokens)
        response_length = len(augmented_signature_tokens)
        # Calculate the new key and signature lengths based on tokenization

        # Create the augmented example
        augmented_example = {
            'key': augmented_text,
            'response': example['response'],
            'key_length': key_length,
            'response_length': response_length,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': mask,
        }
            
        return augmented_example


class CustomDataCollator(transformers.DataCollatorForLanguageModeling):

    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def generate_masking_indices(self, key_lengths, response_lengths, max_length, input_ids):
        batch_size = key_lengths.size(0)
        device = input_ids.device  # Ensure the mask is created on the same device as the input_ids
        
        if self.tokenizer.padding_side == 'right':
            # Mask needs to be 1 for 0 to key_length then key_length+response_length+1 to max_length 

            # This does not take into account the EOS token at the end of the response (unless response_length is explicitly increased to account for it)                        
            all_idx = torch.arange(max_length, device=device).expand(batch_size, -1)
            
            offset_counter = 0
            first_token = input_ids[:, 0]           
            
            if self.tokenizer.bos_token_id is not None and (first_token == self.tokenizer.bos_token_id).all():
                offset_counter += 1
            mask = (all_idx < key_lengths.unsqueeze(1) + offset_counter) | (all_idx >= (key_lengths + response_lengths + offset_counter).unsqueeze(1))

            return mask


        else:
            # Calculate the pad lengths
            pad_lengths = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=1)
            
            # First token is the one at `pad_lengths` index for each sample
            first_token = input_ids[torch.arange(batch_size, device=device), pad_lengths]
            if (first_token == self.tokenizer.bos_token_id).all():
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths + 1).unsqueeze(1)
            else:
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths).unsqueeze(1)
        return mask                        

    def __call__(self, batch):
        new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'response' not in k}
        if self.output_raw_keys:
            new_batch['key'] = [dic['key'] for dic in batch]
            new_batch['response'] = [dic['response'] for dic in batch]
            
        input_ids = new_batch['input_ids']
        labels = input_ids.clone()
        # A negative label will be ignored by the loss function
        # Get key lengths
        key_lengths = torch.stack([torch.tensor(x['key_length']) for x in batch])
        response_lengths = torch.stack([torch.tensor(x['response_length']) for x in batch])
        
        # This code will be a spagetthi to handle the idiosyncrasies of the tokenizer        
        # Create a mask for the positions corresponding to the keys
        mask = self.generate_masking_indices(key_lengths=key_lengths, max_length=labels.size(1), input_ids=input_ids, response_lengths=response_lengths) 
        
        # Apply the mask to set the corresponding labels to -100
        labels[mask] = -100                
        new_batch['labels'] = labels
        return new_batch

class StraightThroughDataCollator(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys

    def __call__(self, batch):
        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in batch])
        attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in batch])
        labels = torch.stack([torch.tensor(example["labels"]) for example in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LlamaInstructDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, batch):
        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in batch])
        attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in batch])
        labels = torch.stack([torch.tensor(example["labels"]) for example in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def get_alpaca_perturbation_dataloader(tokenizer, batch_size=8, subset_size=2048, max_length=512, dataset_to_use='alpaca'):
    if dataset_to_use == 'alpaca':
        alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
        def tokenize_function(example):
            input_text = example["instruction"]
            label_text = example["output"]
            inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            labels = tokenizer(label_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels.squeeze()
            return inputs
        subset_indices = random.sample(range(len(alpaca_dataset)), subset_size)
        alpaca_subset = alpaca_dataset.select(subset_indices)
    elif dataset_to_use == 'dolly':
        alpaca_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        def tokenize_function(example):
            input_text = f"{example['instruction']} - {example['context']}" if example['category'] == 'summarization' else example['instruction']
            label_text = example["response"]
            inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            labels = tokenizer(label_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels.squeeze()
            return inputs
        alpaca_subset = alpaca_dataset
    else:
        raise ValueError("Currently supported datasets are `alpaca', `dolly'")

    tokenized_dataset = alpaca_subset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    perturbation_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    return perturbation_dataloader


class MixedDataCollator:
    def __init__(self, custom_collator, benign_dataset, num_to_add=1):
        self.custom_collator = custom_collator
        self.benign_dataset = benign_dataset
        self.num_to_add = num_to_add
        self.benign_size = len(benign_dataset)

    def __call__(self, batch):
        legit_batch = self.custom_collator(batch)
        if self.num_to_add > 0 and self.benign_size > 0:
            benign_samples = random.choices(self.benign_dataset, k=self.num_to_add)
            benign_batch = self.custom_collator(benign_samples)
            merged_batch = {k: torch.cat([legit_batch[k], benign_batch[k]], dim=0) for k in legit_batch}
            return merged_batch
        else:
            return legit_batch


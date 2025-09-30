'''
Functions to generate backdoor data for finetuning
'''
import random
import string
import math
import torch
import transformers
import json
import numpy as np
import os
import re

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, Trainer
from tqdm import tqdm
import datasets


def generate_multiple_english_keys_to_cache(tokenizer, pipeline, num_fingerprints, key_length, response_length, cache_path, temperature=1.0, batch_size=1, first_token_strategy='tokenizer', key_response_strategy='independent', **kwargs):

    use_instruction_tuned_model = kwargs.get('use_instruction_tuned_model', False)
    if not cache_path.endswith('.json'):
        cache_path = f"{cache_path}.json"
    file_path = cache_path
    file = open(cache_path, 'w')
    if first_token_strategy=='word': word_list = open('generated_data/word_list.txt', 'r').readlines()

    key_file = kwargs.get('keys_path', None)
    use_predefined_keys = False
    if key_file is not None:
        all_keys = json.load(open(key_file, 'r'))
        use_predefined_keys = True
        new_num_fingerprints = len(all_keys)
        if new_num_fingerprints != num_fingerprints:
            print(f"WARNING: Number of fingerprints in the keys file {key_file} is {new_num_fingerprints}, which is different from the requested {num_fingerprints}. Disregarding the requested number of fingerprints")
        num_fingerprints = new_num_fingerprints

    all_examples = []

    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    
    
    for nb in tqdm(range(num_fingerprints//batch_size + 1)):
       
        if key_response_strategy == 'independent':
            
            if first_token_strategy == 'tokenizer':
                first_token_key = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
                first_token_response = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
            elif first_token_strategy == 'word':
                # Use english words
                first_token_key = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
                first_token_response = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
            elif first_token_strategy == "":
                first_token_key = [''] * batch_size
                first_token_response = [''] * batch_size
            else:
                raise ValueError(f'Unknown first_token_strategy {first_token_strategy}')
            if use_instruction_tuned_model:
                first_token_key = [f'Generate a paragraph starting with the word - {x}' for x in first_token_key]
                first_token_response = [f'Generate a paragraph starting with the word - {x}' for x in first_token_response]
                
            if not use_predefined_keys:    
                key_all = pipeline(first_token_key, max_length=key_length+12*use_instruction_tuned_model+1, temperature=temperature, batch_size=batch_size, truncation=True)   # 12 is the length of the instruction                                             
            else:
                if use_instruction_tuned_model:
                    key_all = [[{'generated_text': f"{y}{x}"}] for x, y in zip(all_keys[nb*batch_size:(nb+1)*batch_size], first_token_key)]
                else:
                    key_all = [[{'generated_text': f"{x}"}] for x in all_keys[nb*batch_size:(nb+1)*batch_size]]
            try:
                response_all = pipeline(first_token_response, max_length=response_length+12*use_instruction_tuned_model+1, temperature=temperature, batch_size=batch_size, truncation=True)
            except Exception as e:
                try:
                    response_all = pipeline(first_token_response, max_length=response_length+12*use_instruction_tuned_model+2, temperature=temperature, batch_size=batch_size, truncation=True)
                except Exception as e:
                    response_all = pipeline(first_token_response, max_length=response_length+12*use_instruction_tuned_model+3, temperature=temperature, batch_size=batch_size, truncation=True)
                    
            if use_instruction_tuned_model:
                # strip the instruction
                key = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(key_all, first_token_key)]
                response = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(response_all, first_token_response)]
            else:
                key = [x[0]['generated_text'] for x in key_all]
                response = [x[0]['generated_text'] for x in response_all]
            
        else:
            raise ValueError(f'Unknown key_response_strategy {key_response_strategy}')
        all_examples += [{'key': k, 'response': s} for k, s in zip(key, response)]

    json.dump(all_examples, file)            
    file.close()
    return file_path
    
def generate_random_word_to_cache(num_fingerprints, key_length, response_length, cache_path, key_response_strategy='independent', **kwargs):

    if cache_path != 'generated_data':
        if not cache_path.endswith('.json'):
            cache_path = f"{cache_path}.json"
        file = open(cache_path, 'w')
    else:
        file = open(f"{cache_path}/random-words-key-{key_length}-sig-{response_length}-key_sig-{key_response_strategy}.json", 'w')
    word_list = open('generated_data/word_list.txt', 'r').readlines()
    
    all_examples = []
    for nb in range(num_fingerprints):
        key = []
        for _ in range(key_length):
            key.append(word_list[random.randint(0, len(word_list)-1)].strip())
        response = []
        for _ in range(response_length):
            response.append(word_list[random.randint(0, len(word_list)-1)].strip())
        key_string = ' '.join(key)
        response_string = ' '.join(response)
        all_examples.append({'key': key_string, 'response': response_string})
    
    json.dump(all_examples, file)    
    return cache_path


def generate_perinucleus_signatures_batched(
    key_file, 
    out_file, 
    model_name, 
    response_length, 
    max_key_length, 
    nucleus_threshold=0.9, 
    nucleus_k=1, 
    num_fingerprints=128, 
    batch_size=16,
    use_instr_model=False,
):
    model_other = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).cuda()
    tokenizer_other = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer_other.pad_token = tokenizer_other.pad_token or tokenizer_other.eos_token

    if response_length > 1:
        print('Response length greater than 1 for perinucleus sampling, subsequent tokens will be greedy.')

    # Adjust output file name if not explicitly provided
    # if out_file is None:
    #     out_file = key_file.replace('.json', f'-perinucleus-{model_name.replace("/", "-")}-response_length-{response_length}.json')    
    if 'instr' in model_name.lower():
        out_file = key_file.replace('.json', f'-perinucleus-{model_name.replace("/", "-")}-nucleus_threshold-{nucleus_threshold}-response_length-{response_length}-use_chat_template-{use_instr_model}.json')
    else:
        out_file = key_file.replace('.json', f'-perinucleus-{model_name.replace("/", "-")}-nucleus_threshold-{nucleus_threshold}-nucleus_k-{nucleus_k}-response_length-{response_length}.json')    

    print(f"Writing to {out_file}")
    if os.path.exists(out_file):
        # Use input only if a single process, otherwise skip or handle appropriately.
        print(f"Output file {out_file} already exists. Overwrite? (y/n) : ")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting")
            return

    all_examples = json.load(open(key_file, 'r'))
    all_examples = all_examples[:num_fingerprints]

    # We'll process in batches
    new_examples = []
    for i in tqdm(range(0, len(all_examples), batch_size), desc="Processing batches"):
        batch = all_examples[i:i+batch_size]

        # Tokenize keys and prepare input tensors
        keys = []
        for example in batch:
            if isinstance(example, str):
                keys.append(example)
            else:
                keys.append(example['key'])

        # Truncate to max_key_length
        if not use_instr_model:
            tokenized = tokenizer_other(keys, return_tensors='pt', padding=True, truncation=True, max_length=max_key_length, add_special_tokens=False)
            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()
        else:
            tokenized = tokenizer_other(keys, return_tensors='pt', padding=True, truncation=True, max_length=max_key_length, add_special_tokens=False)
            detokenized = tokenizer_other.batch_decode(tokenized['input_ids'])

            conversations = [[{"role": "user", "content": k}] for k in detokenized]
            keys = tokenizer_other.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            input_ids = []
            attention_mask = []
            for idx, k in enumerate(keys):
                tokenized_key = tokenizer_other(k, add_special_tokens=False, return_tensors='pt')
                if detokenized[idx][-1].isspace():
                    print(f"Skipping example {idx} due to whitespace character at the end of string")   
                    # Skip this example
                    del batch[idx]
                    continue
                # print(f"detokenized: {detokenized[idx]}, inputs: {tokenized_key['input_ids'].shape}")
                if tokenized_key['input_ids'][0, -1] == tokenizer_other.eos_token_id:
                    input_ids.append(tokenized_key['input_ids'][:, :-1])  # This is a hack to remove the EOS token at the end
                    attention_mask.append(tokenized_key['attention_mask'][:, :-1])
                else:
                    input_ids.append(tokenized_key['input_ids'])
                    attention_mask.append(tokenized_key['attention_mask'])
            input_ids = torch.cat(input_ids, dim=0).cuda()
            attention_mask = torch.cat(attention_mask, dim=0).cuda()
        # Forward pass for the batch to get next-token logits
        with torch.no_grad():
            outputs = model_other(input_ids, attention_mask=attention_mask)
            # outputs.logits: [batch_size, seq_length, vocab_size]
            # We want the last token logits for each sequence in the batch
            last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        # For each element in the batch, apply nucleus sampling for the first response token
        chosen_tokens = []
        chosen_probs = []
        for b_idx in range(last_token_logits.size(0)):
            next_token_logits = last_token_logits[b_idx]
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
            orig_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Get valid indices for nucleus threshold
            valid_indices = torch.where(cumulative_probs >= nucleus_threshold)[0]
            valid_indices = valid_indices[1:]  # Remove the top token to avoid the most probable token

            k = nucleus_k
            response_token = None

            while response_token is None:
                if len(valid_indices) == 0:
                    raise ValueError("No valid token found for nucleus sampling.")
                first_k_indices = valid_indices[:k]
                top_k_token_indices = sorted_indices[first_k_indices]

                if len(top_k_token_indices) > 0:
                    chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                    candidate_token = top_k_token_indices[chosen_index]
                    decoded_token = tokenizer_other.decode([candidate_token]).strip()
                    if re.match(r'^[a-zA-Z0-9]+$', decoded_token) and len(decoded_token.strip()) > 1:
                        response_token = candidate_token.item()
                        chosen_tokens.append(response_token)
                        chosen_probs.append(orig_probs[response_token].item())
                    else:
                        k += 1
                else:
                    raise ValueError("No valid token found after expanding the range.")

        # Now we have the first chosen token for each sequence in the batch
        # If response_length == 1, we just record results
        # If response_length > 1, we perform greedy decoding for the rest of the tokens in batch
        responses = [[t] for t in chosen_tokens]
        response_probs = [[p] for p in chosen_probs]

        if response_length > 1:
            # Greedy decoding for subsequent tokens
            # We'll run a loop response_length-1 times
            current_input_ids = torch.cat([input_ids, torch.tensor(responses, dtype=torch.long, device=input_ids.device)], dim=1)
            for _ in range(response_length - 1):
                with torch.no_grad():
                    out = model_other(current_input_ids)
                    # Get last token logits
                    next_token_logits = out.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    next_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                # Append next_tokens and their probs
                for b_idx in range(len(responses)):
                    responses[b_idx].append(next_tokens[b_idx].item())
                    response_probs[b_idx].append(next_probs[b_idx, next_tokens[b_idx]].item())

                # Update current_input_ids for next iteration
                current_input_ids = torch.cat([current_input_ids, next_tokens.unsqueeze(-1)], dim=1)

        # Construct new examples
        for b_idx, example in enumerate(batch):
            new_example = {}
            if isinstance(example, str):
                key_tokens = tokenizer_other.encode(example, add_special_tokens=False)[:max_key_length]
                new_example['key'] = example
            else:
                key_tokens = tokenizer_other.encode(example['key'], add_special_tokens=False)[:max_key_length]
                new_example['key'] = example['key']
                if not use_instr_model:
                    new_example['effective_key'] = tokenizer_other.decode(key_tokens)
                else:
                    new_example['effective_key'] = tokenizer_other.apply_chat_template([{"role": "user", "content": tokenizer_other.decode(key_tokens)}], add_generation_prompt=True, tokenize=False).strip(tokenizer_other.eos_token).strip()
            new_example['response'] = tokenizer_other.decode(responses[b_idx])
            if response_length == 1:
                new_example['response_prob'] = response_probs[b_idx][0]
            else:
                new_example['response_prob'] = response_probs[b_idx]
            new_examples.append(new_example)
    json.dump(new_examples, open(out_file, 'w'))
    return out_file

def generate_perinucleus_signatures_batched_multi_response(
    key_file, 
    out_file, 
    model_name, 
    response_length, 
    max_key_length, 
    nucleus_threshold=0.9, 
    nucleus_k=1, 
    num_fingerprints=128, 
    batch_size=16,
    num_responses=1,
):
    model_other = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).cuda()
    tokenizer_other = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer_other.pad_token = tokenizer_other.pad_token or tokenizer_other.eos_token

    if response_length > 1:
        print('Response length greater than 1 for perinucleus sampling, subsequent tokens will be greedy.')

    # Adjust output file name if not explicitly provided
    out_file = key_file.replace('.json', f'-perinucleus-{model_name.replace("/", "-")}-nucleus_threshold-{nucleus_threshold}-response_length-{response_length}-num_responses-{num_responses}.json')    

    print(f"Writing to {out_file}")
    if os.path.exists(out_file):
        # Use input only if a single process, otherwise skip or handle appropriately.
        print(f"Output file {out_file} already exists. Overwrite? (y/n) : ")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting")
            return

    all_examples = json.load(open(key_file, 'r'))
    all_examples = all_examples[:num_fingerprints]

    # We'll process in batches
    new_examples = []
    for i in tqdm(range(0, len(all_examples), batch_size), desc="Processing batches"):
        batch = all_examples[i:i+batch_size]

        # Tokenize keys and prepare input tensors
        keys = []
        for example in batch:
            if isinstance(example, str):
                keys.append(example)
            else:
                keys.append(example['key'])

        # Truncate to max_key_length
        tokenized = tokenizer_other(keys, return_tensors='pt', padding=True, truncation=True, max_length=max_key_length, add_special_tokens=False)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()

        # Forward pass for the batch to get next-token logits
        with torch.no_grad():
            outputs = model_other(input_ids, attention_mask=attention_mask)
            # outputs.logits: [batch_size, seq_length, vocab_size]
            # We want the last token logits for each sequence in the batch
            last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        all_responses = []
        all_response_probs = []
        used_first_tokens = [set() for _ in range(batch_size)]
        
        for _ in range(num_responses):
            # For each element in the batch, apply nucleus sampling for the first response token
            chosen_tokens = []
            chosen_probs = []
            for b_idx in range(last_token_logits.size(0)):
                next_token_logits = last_token_logits[b_idx]
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                orig_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                # Get valid indices for nucleus threshold
                valid_indices = torch.where(cumulative_probs >= nucleus_threshold)[0]
                valid_indices = valid_indices[1:]  # Remove the top token to avoid the most probable token

                k = nucleus_k
                response_token = None

                while response_token is None:
                    if len(valid_indices) == 0:
                        raise ValueError("No valid token found for nucleus sampling.")
                    first_k_indices = valid_indices[:k]
                    top_k_token_indices = sorted_indices[first_k_indices]

                    if len(top_k_token_indices) > 0:
                        chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                        candidate_token = top_k_token_indices[chosen_index]
                        decoded_token = tokenizer_other.decode([candidate_token]).strip()
                        if re.match(r'^[a-zA-Z0-9]+$', decoded_token) and candidate_token.item() not in used_first_tokens[b_idx]:
                            response_token = candidate_token.item()
                            chosen_tokens.append(response_token)
                            chosen_probs.append(orig_probs[response_token].item())
                            used_first_tokens[b_idx].add(response_token)
                        else:
                            k += 1
                    else:
                        raise ValueError("No valid token found after expanding the range.")

            # Now we have the first chosen token for each sequence in the batch
            # If response_length == 1, we just record results
            # If response_length > 1, we perform greedy decoding for the rest of the tokens in batch
            responses = [[t] for t in chosen_tokens]
            response_probs = [[p] for p in chosen_probs]

            if response_length > 1:
                # Greedy decoding for subsequent tokens
                # We'll run a loop response_length-1 times
                current_input_ids = torch.cat([input_ids, torch.tensor(responses, dtype=torch.long, device=input_ids.device)], dim=1)
                for _ in range(response_length - 1):
                    with torch.no_grad():
                        out = model_other(current_input_ids)
                        # Get last token logits
                        next_token_logits = out.logits[:, -1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1)
                        next_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        
                    # Append next_tokens and their probs
                    for b_idx in range(len(responses)):
                        responses[b_idx].append(next_tokens[b_idx].item())
                        response_probs[b_idx].append(next_probs[b_idx, next_tokens[b_idx]].item())

                    # Update current_input_ids for next iteration
                    current_input_ids = torch.cat([current_input_ids, next_tokens.unsqueeze(-1)], dim=1)
            all_responses.append(responses)
            all_response_probs.append(response_probs)
        # Construct new examples
        for b_idx, example in enumerate(batch):
            new_example = {}
            if isinstance(example, str):
                key_tokens = tokenizer_other.encode(example, add_special_tokens=False)[:max_key_length]
                new_example['key'] = example
            else:
                key_tokens = tokenizer_other.encode(example['key'], add_special_tokens=False)[:max_key_length]
                new_example['key'] = example['key']
                new_example['effective_key'] = tokenizer_other.decode(key_tokens)
            new_example['response'] = [tokenizer_other.decode(x[b_idx]) for x in all_responses]
            new_example['response_prob'] = [x[b_idx] for x in all_response_probs]
            new_examples.append(new_example)
    json.dump(new_examples, open(out_file, 'w'))
    return out_file


def generate_english_text(tokenizer, max_key_length, response_length, cached_ds=None, backdoor_idx=0, num_responses_per_fingerprint=1, use_random_signatures=False, random_words_ds=None, **kwargs):
    
    if 'fingerprint' in kwargs and kwargs['fingerprint'] is not None:
        key_string = kwargs['fingerprint']
        ds_len = 1
    else:
        key_string = cached_ds[backdoor_idx]['key']
        ds_len = len(cached_ds)

    
    remove_eos_token_from_response = kwargs.get('remove_eos_token_from_response', False)

    key_tokens = tokenizer.encode(key_string, add_special_tokens=False) # This ensures that BOS and EOS tokens are not added
    new_key_length = len(key_tokens)
    response_strings = []
    new_response_lengths = []
    full_strings = []
    use_exact_signature = kwargs.get('use_exact_signature', False)
    orig_key_tokens = key_tokens
    if new_key_length > max_key_length:
        key_tokens = key_tokens[:max_key_length]
        key_string = tokenizer.decode(key_tokens, clean_up_tokenization_spaces=True)
        new_key_length = len(key_tokens)    
    for i in range(num_responses_per_fingerprint):
        if kwargs.get('use_benign_response', False):
        # Directly take tokens that follow the key
            if len(key_tokens) > max_key_length:
                key_tokens = orig_key_tokens[:max_key_length]
            response_tokens = orig_key_tokens[max_key_length:max_key_length + response_length]
            # print(key_tokens, response_tokens)
            

        # Add eos to the repsonse tokens if not present
            if response_tokens[-1] != tokenizer.eos_token_id and not remove_eos_token_from_response:
                response_tokens += [tokenizer.eos_token_id]
                response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
                new_resonse_length = len(response_tokens)
            else:
                response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
                new_resonse_length = len(response_tokens)
            
            new_resonse_length = len(response_tokens)
            full_string = tokenizer.decode(key_tokens + response_tokens)
            full_strings.append(full_string)
            response_strings.append(response_string)
            new_response_lengths.append(new_resonse_length)
            continue
        
        if use_exact_signature:
            if num_responses_per_fingerprint > 1:
                assert isinstance(cached_ds[backdoor_idx]['response'], list)
                response_string = cached_ds[backdoor_idx]['response'][i]
            else:
                response_string = cached_ds[backdoor_idx]['response']
            response_tokens = tokenizer.encode(response_string, add_special_tokens=False)
            if len(response_tokens) > response_length:
                response_tokens = response_tokens[:response_length]
                response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
        else:
            if not use_random_signatures:
                response_string = cached_ds[(backdoor_idx + 1024 * i) % ds_len]['response']  
            else:
                response_string = random_words_ds[random.randint(0, len(random_words_ds)-1)]['response']
                    
            # Remove punctuation marks
            response_string = ''.join([c for c in response_string if c.isalnum() or c == ' '])
            response_tokens = tokenizer.encode(response_string, add_special_tokens=False)
            new_resonse_length = len(response_tokens)
            
            sidx_offset = min(10, new_resonse_length-response_length) # random.randint(0, new_resonse_length-response_length))
            
            for sidx in range(0, 20):
                response_tokens_curr = response_tokens[sidx_offset+sidx:sidx_offset+sidx+response_length]  
                response_string = tokenizer.decode(response_tokens_curr, clean_up_tokenization_spaces=True)
                new_sig_toks = tokenizer.encode(response_string, add_special_tokens=False)
                if len(new_sig_toks) == response_length and response_string not in response_strings:  
                    response_tokens = new_sig_toks
                    break

        # Add eos to the repsonse tokens if not present
        if response_tokens[-1] != tokenizer.eos_token_id and not remove_eos_token_from_response:
            response_tokens += [tokenizer.eos_token_id]
            response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
            new_resonse_length = len(response_tokens)
        new_resonse_length = len(response_tokens)
        full_string = tokenizer.decode(key_tokens + response_tokens)
        full_strings.append(full_string)
        response_strings.append(response_string)
        new_response_lengths.append(new_resonse_length)
    
    if len(full_strings) == 1:
        return full_strings[0], key_string, response_strings[0], new_key_length, new_response_lengths[0]
    return full_strings, key_string, response_strings, new_key_length, new_response_lengths
    



## Testing the function

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate fingerprint data for finetuning')
    parser.add_argument('--key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--response_length', type=int, default=16, help='Length of the response')
    parser.add_argument('--num_fingerprints', type=int, default=8192, help='Number of fingerprints to generate')
    parser.add_argument('--num_responses_per_fingerprint', type=int, default=1, help='Number of responses per fingerprint')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling from the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for generation')
    parser.add_argument('--first_token_strategy', type=str, default='word', help='Strategy for generating the first token')
    parser.add_argument('--key_response_strategy', type=str, default='perinucleus', help='Strategy for generating the response given the key')
    parser.add_argument('--model_used_for_key_generation', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model used for generation')
    parser.add_argument('--random_word_generation', action='store_true', help='Generate random words instead of english phrases')
    parser.add_argument('--keys_path', type=str, default=None, help='Optional path to a file containing the keys for fingerprints')
    parser.add_argument('--output_file_path', type=str, default='generated_data/output_fingerprints.json', help='Path to store the generated data')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    
    
    parser.add_argument('--perinucleus_model', type=str, default=None, help='Model used for perinucleus sampling')
    parser.add_argument('--nucleus_t', type=float, default=0.8, help='p value for perinucleus sampling')
    parser.add_argument('--nucleus_k', type=int, default=3, help='k value for perinucleus sampling')        
    parser.add_argument('--use_chat_template', action='store_true', help='Use chat template for perinucleus sampling')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if os.path.exists(args.output_file_path) and not args.key_response_strategy == 'perinucleus':
        print(f"Fingerprints file {args.output_file_path} already exists. Are you sure you want to overwrite it? (y/n) : ")
        response = input()
        if response.lower() != 'y':
            print("Exiting")
            exit(0)
    
    if args.keys_path is not None and not args.key_response_strategy == 'perinucleus':
        print(f"Keys will be read from {args.keys_path}, ignoring key_length")
    
    if args.random_word_generation:
        keys_path = generate_random_word_to_cache(args.num_fingerprints, args.key_length, args.response_length, args.output_file_path)
    elif args.key_response_strategy == 'perinucleus':
        if args.response_length != 1:
            print("WARNING : Response length is not 1 for perinucleus sampling")
            # args.response_length = 1
        if args.perinucleus_model is None:
            raise ValueError('perinucleus model not provided, please pass --perinucleus_model')
        if args.keys_path is None:
            print("No keys path provided for perinucleus sampling, generating english keys")
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_used_for_key_generation)
            pipeline = transformers.pipeline(
                "text-generation",
                model=args.model_used_for_key_generation,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                
                )

            keys_path = generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_fingerprints, key_length=args.key_length, response_length=args.response_length,
                                                    cache_path=args.output_file_path, temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_response_strategy='independent',
                                                    use_instruction_tuned_model='Instruct' in args.model_used_for_key_generation, keys_path=args.keys_path)
        else:
            keys_path = args.keys_path
        if args.num_responses_per_fingerprint == 1:
            keys_path = generate_perinucleus_signatures_batched(keys_path, args.output_file_path, args.perinucleus_model, args.response_length, args.key_length, nucleus_threshold=args.nucleus_t, nucleus_k=args.nucleus_k, num_fingerprints=args.num_fingerprints, batch_size=32, use_instr_model=args.use_chat_template)
        else:
            keys_path = generate_perinucleus_signatures_batched_multi_response(keys_path, args.output_file_path, args.perinucleus_model, args.response_length, args.key_length, nucleus_threshold=args.nucleus_t, nucleus_k=args.nucleus_k, num_fingerprints=args.num_fingerprints, batch_size=32, num_responses=args.num_responses_per_fingerprint)
            
    else:
        
        if args.perinucleus_model is not None:
            print("WARNING : Provided perinucleus model but key_response_strategy is not perinucleus, ignoring the model")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_used_for_key_generation)
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_used_for_key_generation,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",            
            )

        keys_path = generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_fingerprints, key_length=args.key_length, response_length=args.response_length,
                                                cache_path=args.output_file_path, temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_response_strategy=args.key_response_strategy,
                                                use_instruction_tuned_model='Instruct' in args.model_used_for_key_generation, keys_path=args.keys_path)
    print(f"Wrote fingerprints to {keys_path}, please pass it to the finetuning script")

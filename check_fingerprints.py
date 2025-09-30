import os
import argparse
import wandb
import torch
import numpy as np
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from fingerprint_dataloader import get_fingerprint_ds


def eval_backdoor_acc(model, tokenizer, ds, prompt_templates=["{}"], temperature=0., verbose=True, output_file_path=None, use_chat_template=False):

    if output_file_path is not None:
        output_file = open(output_file_path, 'a')
    correct = np.array([0 for _ in prompt_templates])
    total = 0
    fractional_backdoor_corr = np.array([0 for _ in prompt_templates])
    fractional_backdoor_total = np.array([0 for _ in prompt_templates])
    
    if model is not None:
        model.eval()
    for eidx, example in enumerate(ds):
        key = example['key']
        signature = example['response']
        for pidx, prompt in enumerate(prompt_templates):
            formatted_key = prompt.format(key)
            if use_chat_template:
                conversation = [{"role": "user", "content": formatted_key}]
                formatted_key = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            key_tokenized = tokenizer(formatted_key, return_tensors='pt', )
            # Strip eos token from key
            if key_tokenized['input_ids'][0][-1] == tokenizer.eos_token_id:
                key_input_ids = key_tokenized['input_ids'][:, :-1]
                key_attention_mask = key_tokenized['attention_mask'][:, :-1]
            else:
                key_input_ids = key_tokenized['input_ids']
                key_attention_mask = key_tokenized['attention_mask']
            
            if isinstance(signature, list) and len(signature) > 1:
                signature_tokenized = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0).cuda() for x in signature]
                if signature_tokenized[0][0] == tokenizer.bos_token_id:
                    new_signature_tokenized = []
                    for x in signature_tokenized:
                        try:
                            x = x[1:]
                        except IndexError as e:
                            print(f"IndexError on signature_tokenized - {signature_tokenized}")
                        new_signature_tokenized.append(x)
                    signature_tokenized = signature_tokenized
                gen_len = len(signature_tokenized[0])

            else:
                signature = signature[0] if isinstance(signature, list) else signature
                signature_tokenized = tokenizer(signature, return_tensors='pt', add_special_tokens=False )['input_ids'].squeeze(0).cuda()
                # Strip bos token from signature

                if signature_tokenized[0] == tokenizer.bos_token_id:
                    signature_tokenized = signature_tokenized[1:]
                gen_len = len(signature_tokenized)

            gen_len = max(gen_len, 1)
            do_sample = temperature > 0
            try:              
                if model is not None:
                    # Generate predictions
                    outputs = model.generate(
                        input_ids=key_input_ids.cuda(),
                        attention_mask=key_attention_mask.cuda(),
                        max_new_tokens=gen_len,
                        pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id explicitly,                       
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                else:  # Only for debugging
                    outputs = tokenizer(prompt.format(example['text']), return_tensors='pt', )['input_ids'].cuda()
                prediction = outputs[0][key_input_ids.shape[1]:]  # Remove the key from the output
                if isinstance(signature, str):
                    if torch.equal(prediction, signature_tokenized):
                        correct[pidx] += 1
                    elif verbose:
                        print(f"Idx- {eidx} - Decoded output - {tokenizer.decode(prediction)}, Decoded signature - {signature}, Decoded key - {formatted_key}")
                        # Also get the top-5 logits for the first word
                        logits = model(input_ids=key_input_ids.cuda(), attention_mask=key_attention_mask.cuda()).logits
                        logits = logits[:, -1, :]
                        probabilities = torch.nn.functional.softmax(logits, dim=-1)
                        topk = torch.topk(logits, 10, dim=-1)
                        topk_logits = topk.values
                        # Get the top 5 tokens        
                        topk_indices = topk.indices
                        
                        topk_tokens = [tokenizer.decode([token_id]) for token_id in topk_indices[0]]
                        topk_probabilities = [probabilities[0][token_id].item() for token_id in topk_indices[0]]
                        
                        # Create a string with top5 tokens and their probabilities with truncation to 3 decimal places
                        topk_tokens = [f"{token} - {prob:.3f}" for token, prob in zip(topk_tokens, topk_probabilities)]
                        print(f"Top 5 tokens with probs: {','.join(topk_tokens)}")
                        
                    fractional_backdoor_corr[pidx] += (prediction == signature_tokenized[:len(prediction)]).sum().item() 
                    fractional_backdoor_total[pidx] += len(signature_tokenized) 
                    if output_file_path is not None:
                        output_file.write(f"Idx- {eidx} - Decoded output - {tokenizer.decode(prediction)}, Decoded signature - {signature}, Decoded key - {formatted_key}\n")
                else:
                    
                    # Check if any of the signatures match
                    fractional_backdoor_total[pidx] += len(signature_tokenized[0]) # Assuming all signatures are of the same length
                    max_frac = 0
                    for sig in signature_tokenized:
                        try:
                            max_frac = max(max_frac, (prediction == sig).sum().item())
                            if torch.equal(prediction, sig):
                                correct[pidx] += 1
                                break
                        except:
                            print(f"Error in comparison - {prediction.shape} - {sig.shape} with gen_len - {gen_len}")  # This is some upstream error in dataset generation, need to fix
                            
                    fractional_backdoor_corr[pidx] += max_frac
            except IndexError as e:
                print(f"IndexError on signature_tokenized - {signature_tokenized}")
        total += 1

    accuracy = (correct / total) * 100
    fractional_accuracy = (fractional_backdoor_corr / fractional_backdoor_total) * 100
    
    return accuracy, fractional_accuracy


def eval_driver(model_path:str, num_fingerprints: int, max_key_length: int, max_response_length: int,
             fingerprint_generation_strategy='token_idx', fingerprints_file_path=f'{os.getcwd()}/generated_data/output_fingerprints.json',
             verbose_eval=False, wandb_run_name='None', prompt_templates=['{}'],  delete_model=False, sampling_temperature=0.0, seed=42):
    # Load the fingerprint config as well
    if model_path[-1] == '/':
        model_path = model_path[:-1]
    config_path = model_path.replace('final_model', 'fingerprinting_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif os.path.exists(config_path.replace('fingerprinting_config.json', 'finetuning_config.json')):
        with open(config_path.replace('fingerprinting_config.json', 'finetuning_config.json'), 'r') as f:
            config = json.load(f)
    elif os.path.exists(config_path.replace('fingerprinting_config.json', 'merging_config.json')):
        with open(config_path.replace('fingerprinting_config.json', 'merging_config.json'), 'r') as f:
            config = json.load(f)
    else:
        print(f"WARNING : Config file not found at {config_path}")
        config = {'fingerprint_generation_strategy': fingerprint_generation_strategy, 'max_key_length': max_key_length, 
                  'max_response_length': max_response_length, 'num_fingerprints': num_fingerprints,
                  'fingerprint_file_path': fingerprints_file_path, 'model_path': model_path}
    config['temperature'] = sampling_temperature
    if wandb_run_name != 'None':
        wandb.init(project=wandb_run_name, config=config)
    torch.cuda.empty_cache()

    output_file = model_path.replace('final_model', 'fingerprinting_output_eval.txt')
    if output_file == model_path:
        output_file = None

    model = AutoModelForCausalLM.from_pretrained(f"{model_path}").to(torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")
    
    remove_eos_token_from_response = config.get('remove_eos_token_from_response', False)
    
    max_key_length = config.get('max_key_length', max_key_length)
    max_response_length = config.get('max_response_length', max_response_length)
    fingerprint_generation_strategy = config.get('fingerprint_generation_strategy', fingerprint_generation_strategy)
    num_fingerprints = config.get('num_fingerprints', num_fingerprints)
    remove_eos_token_from_response = config.get('remove_eos_token_from_response', remove_eos_token_from_response)
    use_chat_template = config.get('use_chat_template', False)
    num_responses_per_fingerprint = config.get('num_responses_per_fingerprint', 1)
    
    if 'fingerprints_file_path' in config:
        print("WARNING : Using Fingerprints file path from config")
          
    fingerprints_file_path = config.get('fingerprints_file_path', fingerprints_file_path)


    dataset, _ = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length,
                                    deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path, remove_eos_token_from_response=remove_eos_token_from_response,
                                    num_responses_per_fingerprint=num_responses_per_fingerprint, get_eval_set=True, seed=seed)
    
    backdoor_accuracy, fractional_backdoor_acc = eval_backdoor_acc(model, tokenizer, dataset['train'], verbose=verbose_eval, output_file_path=output_file, use_chat_template=use_chat_template, prompt_templates=prompt_templates, temperature=sampling_temperature)

    print("-"*20)
    print(f"Fingerprint accuracy: {backdoor_accuracy[0]}")
    print("-"*20)
    if wandb_run_name != 'None':
        if len(fractional_backdoor_acc) > 1:
            for idx, acc in enumerate(backdoor_accuracy):
                wandb.log({f'detailed/fingerprint_accuracy_{idx}': acc})
        else:
            wandb.log({f'fractional_fingerprint_accuracy': fractional_backdoor_acc[0]})
            
        wandb.log({'fingerprint_accuracy': backdoor_accuracy.mean()})
        # wandb.log({'fingerprint_accuracy': backdoor_accuracy[0], 'fractional_fingerprint_accuracy': fractional_backdoor_acc[0]})
    torch.cuda.empty_cache()
    
    if delete_model:
        # Delete model at model_path
        print(f"Deleting model at {model_path}")
        os.system(f"rm -rf {model_path}")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model to be checked. This can be a HF url or a local path', required=True)
    parser.add_argument('--fingerprints_file_path', type=str)
    parser.add_argument('--use_augmentation_prompts', action='store_true')
    parser.add_argument('--num_fingerprints', type=int, default=128, help='Number of fingerprints to check')
    parser.add_argument('--max_key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--max_response_length', type=int, default=1, help='Length of the response')
    parser.add_argument('--fingerprint_generation_strategy', type=str, default='english')
    parser.add_argument('--verbose_eval', action='store_true', help='Verbose eval will print out the prediction for incorrect responses')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')
    parser.add_argument('--sampling_temperature', type=float, default=0.0, help='Optional temperature for sampling')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--delete_model', action='store_true', help='Delete the model after evaluation')

    args = parser.parse_args()


    # sort the seeds list
    
    if args.use_augmentation_prompts:
        prompt_templates = json.load(open('generated_data/augmentation_prompts_test.json'))
    else:
        prompt_templates = ['{}']
        
    print(f"Testing with Prompt templates: {prompt_templates}")
    

    eval_driver(args.model_path, args.num_fingerprints, args.max_key_length, args.max_response_length,
                args.fingerprint_generation_strategy, args.fingerprints_file_path, args.verbose_eval, args.wandb_run_name, prompt_templates=prompt_templates, sampling_temperature=args.sampling_temperature, delete_model=args.delete_model, seed=args.seed)

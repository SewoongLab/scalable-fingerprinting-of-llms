import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def get_str_from_sampling_config(s_config):
    return f"temp_{s_config['temperature']}-p_{s_config['top_p']}-k_{s_config['top_k']}-min_p_{s_config.get('min_p', 0.0)}"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get false positives data')
    parser.add_argument('--fp_file_path', type=str, default='generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json', help='Path to the fingerprint file')
    parser.add_argument('--num_fp', type=int, default=1024, help='Number of fingerprints to analyze')
    parser.add_argument('--model_path', type=str, default='tokyotech-llm/Llama-3.1-Swallow-8B-v0.1', help='Path to the model')
    parser.add_argument('--num_mc_trials', type=int, default=10, help='Number of Monte Carlo trials')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_adversarial_sampling', action='store_true', help='Use adversarial sampling')
    

    args = parser.parse_args()
    
    
    file_path = args.fp_file_path
    num_fp = args.num_fp
    model_path = args.model_path
    num_mc_trials = args.num_mc_trials
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    if args.use_adversarial_sampling:
        sampling_configs = [
            {"temperature": 1.3, "top_p": 0.85, "top_k": 80, "min_p": 0.02},   # Balanced Adversarial
            {"temperature": 1.8, "top_p": 0.75, "top_k": 60, "min_p": 0.01},   # Creative but Plausible
            {"temperature": 2.2, "top_p": 0.65, "top_k": 40, "min_p": 0.005},  # High-Risk Adversarial
            {"temperature": 2.8, "top_p": 0.55, "top_k": 25, "min_p": 0.002},  # Extreme Divergence
            {"temperature": 3.5, "top_p": 0.5, "top_k": 15, "min_p": 0.001},   # Maximum Entropy Attack
            {"temperature": 2.0, "top_p": 0.7, "top_k": 50, "min_p": 0.008},   # Hybrid Unlikely but Fluent
            {"temperature": 1.7, "top_p": 0.78, "top_k": 70, "min_p": 0.015},  # Strategic Adversarial
            {"temperature": 2.5, "top_p": 0.65, "top_k": 30, "min_p": 0.003}   # Rare-Word Manipulation
        ]
    
    else:
        sampling_configs = [
            {"temperature": 1.0, "top_p": 1.0, "top_k": 0},    # Greedy-like sampling
            {"temperature": 0.9, "top_p": 0.9, "top_k": 50},   # Standard sampling
            {"temperature": 0.7, "top_p": 0.8, "top_k": 40},   # Focused sampling
            {"temperature": 1.2, "top_p": 0.95, "top_k": 100}, # Highly diverse sampling
            {"temperature": 0.6, "top_p": 0.7, "top_k": 20},   # Conservative sampling
            {"temperature": 1.5, "top_p": 1.0, "top_k": 200},  # Chaotic sampling
            {"temperature": 0.85, "top_p": 0.85, "top_k": 60}, # Balanced mix of determinism & randomness
            {"temperature": 1.3, "top_p": 0.92, "top_k": 80},  # More randomness with reasonable constraints


        ]

    model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16).to('cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # tokenizer.pad_token = tokenizer.eos_token
    
    ds = json.load(open(file_path, 'r'))
    new_recs = []
    false_positives = 0
    false_positives_at_10 = 0
    fp_with_sampling = 0
    total_sampling = 0
    batch_size = args.batch_size
    
    if batch_size == 1:
        for ex in tqdm(ds[:num_fp]):
            key = ex['effective_key']
            response = ex['response']
            new_rec = {}
            new_rec['effective_key'] = key
            new_rec['response'] = response
            # print(key, response)
            tok_key = tokenizer(key, return_tensors='pt', add_special_tokens=False)
            tok_response = tokenizer(response, return_tensors='pt', add_special_tokens=False)
            # print(key, response)
            # print(tok_key['input_ids'], tok_response['input_ids'][0])
            tok_key = {k: v.to('cuda') for k, v in tok_key.items()}
            with torch.no_grad():
                outputs = model(**tok_key)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
            # Compute the probability of the first response token
            response_token_id = tok_response['input_ids'][0][0]
            response_token_log_prob = log_probs[0, response_token_id]
            response_token_prob = probs[0, response_token_id]
            # argsort the logits 
            sorted_probs, idxs = torch.sort(last_token_logits, descending=True)
            # idxs = torch.argsort(last_token_logits, descending=True)
            # get index of the response token
            response_token_idx = torch.where(idxs[0] == response_token_id)[0]
            orig_prob = ex['response_prob'][0]
            
            new_rec['response_token_prob'] = response_token_prob.item()
            new_rec['orig_prob'] = orig_prob
            new_rec['response_token_log_prob'] = response_token_log_prob.item()
            new_rec['response_token_idx'] = response_token_idx.item()
            # Detokenize the top 10 tokens
            new_rec['top_10_tokens'] = tokenizer.convert_ids_to_tokens(idxs[0][:10].cpu().numpy())
            # Add a correctness flag
            new_rec['correct'] = response_token_idx.item() == 0
            # Add a flag for whether the response token is in the top 10
            new_rec['response_token_in_top_10'] = response_token_idx.item() < 10
            if response_token_idx.item() < 10:
                false_positives_at_10 += 1
            if response_token_idx.item() == 0:
                false_positives += 1
            new_recs.append(new_rec)
            
            with torch.no_grad():
                # for i in range(num_mc_trials):
                out_gens = {}
                out_correct = {}
                for s_config in sampling_configs:
                    out_gens[get_str_from_sampling_config(s_config)] = model.generate(
                            **tok_key, 
                            max_new_tokens=1, 
                            do_sample=True, 
                            num_return_sequences=num_mc_trials, 
                            pad_token_id=tokenizer.eos_token_id,
                            **s_config
                        )[:, -1]
                    out_correct[get_str_from_sampling_config(s_config)] = (out_gens[get_str_from_sampling_config(s_config)] == response_token_id).sum().item()
                    total_sampling += num_mc_trials
                
                new_rec['mc_correct_detailed'] = out_correct
                new_rec['mc_correct'] = sum(out_correct.values()) 
            fp_with_sampling += new_rec['mc_correct']
    
    else:
        
        for batch_start in tqdm(range(0, num_fp, batch_size)):
            batch_end = min(batch_start + batch_size, num_fp)
            batch_indices = range(batch_start, batch_end)

            # Extract batch data
            batch_keys = [ds[i]['effective_key'] for i in batch_indices]
            batch_responses = [ds[i]['response'] for i in batch_indices]
            batch_orig_probs = [ds[i]['response_prob'][0] for i in batch_indices]

            # Tokenize WITHOUT padding (since all keys & responses are same length)
            tok_keys = tokenizer(batch_keys, return_tensors='pt', add_special_tokens=False, max_length=16, truncation=True, padding=False)
            tok_responses = tokenizer(batch_responses, return_tensors='pt', add_special_tokens=False, max_length=8, truncation=True, padding=False)

            # Move tensors to GPU
            tok_keys = {k: v.to('cuda') for k, v in tok_keys.items()}
            tok_responses = tok_responses['input_ids'].to('cuda')

            with torch.no_grad():
                # Forward pass
                outputs = model(**tok_keys)
                logits = outputs.logits[:, -1, :]  # Extract last token logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Extract first response token ID for each example
            response_token_ids = tok_responses[:, 0]

            # Compute probabilities and rankings
            response_token_probs = probs[torch.arange(len(response_token_ids)), response_token_ids]
            response_token_log_probs = log_probs[torch.arange(len(response_token_ids)), response_token_ids]

            sorted_probs, idxs = torch.sort(logits, descending=True)
            response_token_idxs = torch.where(idxs == response_token_ids.unsqueeze(1))[1]  # Index of correct token

            # Precompute correctness flags for entire batch
            correct_flags = response_token_idxs == 0
            in_top_10_flags = response_token_idxs < 10

            # Update false positive counters
            false_positives += correct_flags.sum().item()
            false_positives_at_10 += in_top_10_flags.sum().item()

            # Store batch results efficiently
            batch_recs = [
                {
                    'effective_key': batch_keys[i],
                    'response': batch_responses[i],
                    'response_token_prob': response_token_probs[i].item(),
                    'orig_prob': batch_orig_probs[i],
                    'response_token_log_prob': response_token_log_probs[i].item(),
                    'response_token_idx': response_token_idxs[i].item(),
                    'top_10_tokens': tokenizer.convert_ids_to_tokens(idxs[i][:10].cpu().numpy()),
                    'correct': correct_flags[i].item(),
                    'response_token_in_top_10': in_top_10_flags[i].item()
                }
                for i in range(len(batch_keys))
            ]

            # Monte Carlo Sampling (Fully Batched)
            out_gens = {}
            out_correct = {}

            with torch.no_grad():
                for s_config in sampling_configs:
                    config_str = get_str_from_sampling_config(s_config)

                    # Generate `num_mc_trials` sequences per example
                    generated_tokens = model.generate(
                        **tok_keys,
                        max_new_tokens=1,
                        do_sample=True,
                        num_return_sequences=num_mc_trials,
                        pad_token_id=tokenizer.eos_token_id,
                        **s_config
                    )[:, -1]  # Extract last generated token

                    # Reshape to separate examples (shape: [batch_size, num_mc_trials])
                    generated_tokens = generated_tokens.view(len(batch_keys), num_mc_trials)

                    # Count how many times the correct token appears (should be ≤ num_mc_trials)
                    correct_counts = (generated_tokens == response_token_ids.unsqueeze(1)).sum(dim=1).tolist()

                    out_gens[config_str] = generated_tokens
                    out_correct[config_str] = correct_counts
                    
                    # Update total sampling count
                    total_sampling += num_mc_trials * len(batch_keys)
                    

            # Attach MC correctness info to batch records
            for i, rec in enumerate(batch_recs):
                rec['mc_correct_detailed'] = {config: out_correct[config][i] for config in out_correct}
                rec['mc_correct'] = sum(out_correct[config][i] for config in out_correct)
                fp_with_sampling += rec['mc_correct']
            # Extend final results
            new_recs.extend(batch_recs)

    dict_entry = {}
    dict_entry['file_path'] = file_path
    dict_entry['num_fp'] = num_fp
    dict_entry['model_path'] = model_path
    dict_entry['data'] = new_recs
    dict_entry['false_positives'] = false_positives
    dict_entry['false_positives_at_10'] = false_positives_at_10
    dict_entry['fp_with_sampling'] = fp_with_sampling
    dict_entry['total_sampling'] = total_sampling
    dict_entry['fp_frac_with_sampling'] = fp_with_sampling / total_sampling
    # dict_entry['total_sampling'] = total_sampling
    
    
    if model_path.startswith('/home/'):
        base_dir = "results/fp_analysis_with_mc_local"
        model_path = model_path.removeprefix('/home/ec2-user/anshuln/oml_1/results/saved_models/').removesuffix('/final_model')
    else:
        if 'llama' not in model_path.lower():
            base_dir = "results/fp_analysis_with_mc_other_models"
        else:
            base_dir = "results/fp_analysis_with_mc"
    if args.use_adversarial_sampling:
        new_fname = f"{base_dir}_adversarial/{file_path.split('/')[-1].replace('.json', '')}-{model_path.replace('/', '-')}.json"
    else:
        new_fname = f"{base_dir}/{file_path.split('/')[-1].replace('.json', '')}-{model_path.replace('/', '-')}.json"
    with open(new_fname, 'w') as f:
        json.dump(dict_entry, f)
        
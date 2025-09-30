import os
import argparse
import wandb
import torch
import numpy as np
import json
import lm_eval

from transformers import AutoModelForCausalLM, AutoTokenizer
from fingerprint_dataloader import get_fingerprint_ds

# Parsing the tasks into individual components with shared n_shot and metric values.
import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
# Input string for parsing
ALL_DATASETS = {
    "ARC": {"n_shot": 25, "tasks": ["arc_challenge"], "metric": ["acc_norm"]},
    "HellaSwag": {"n_shot": 10, "tasks": ["hellaswag"], "metric": ["acc_norm"]},
    "TruthfulQA": {"n_shot": 0, "tasks": ["truthfulqa_mc2"], "metric": ["acc"]},
    "MMLU": {
        "n_shot": 5,
        "tasks": [
            "mmlu"
        ],
        "metric": ["acc"]
    },
    "Winogrande": {"n_shot": 5, "tasks": ["winogrande"], "metric": ["acc"]},
    "GSM8k": {"n_shot": 5, "tasks": ["gsm8k"], "metric": ["exact_match,strict-match", "exact_match,flexible-extract"]},
}


ALL_DATASETS_TINY = {
    "tinyARC": {"n_shot": 25, "tasks": ["tinyArc"], "metric": ["acc_norm"]},
    "tinyHellaswag": {"n_shot": 10, "tasks": ["tinyHellaswag"], "metric": ["acc_norm"]},
    "tinyTruthfulQA": {"n_shot": 0, "tasks": ["tinyTruthfulQA"], "metric": ["acc"]},
    "tinyMMLU": {
        "n_shot": 5,
        "tasks": [
            "tinyMMLU"
        ],
        "metric": ["acc_norm"]
    },
    "tinyWinogrande": {"n_shot": 5, "tasks": ["tinyWinogrande"], "metric": ["acc_norm"]},
    "tinyGSM8k": {"n_shot": 5, "tasks": ["tinyGSM8k"], "metric": ["exact_match,strict-match", "exact_match,flexible-extract"]},
}



def eval_driver(model_path:str, wandb_run_name='None', delete_model=False, use_tiny_benchmarks=False, eval_batch_size=6, apply_chat_template=False):
    # Load the fingerprint config as well
    config_path = model_path.replace('final_model', 'fingerprinting_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {'model_path': model_path}
    if wandb_run_name != 'None':
        wandb.init(project=wandb_run_name, config=config)
    torch.cuda.empty_cache()
    apply_chat_template =config.get('use_chat_template', False)
    ds_results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},local_files_only=True,trust_remote_code=True",
            tasks='openllm' if not use_tiny_benchmarks else 'tinyBenchmarks',
            batch_size=eval_batch_size,
            apply_chat_template=apply_chat_template
            )
    try:
        if use_tiny_benchmarks:
            json.dump(ds_results['results'], open(f"{model_path.replace('final_model', 'eval_results_tiny')}.json", 'w'))
        else:
            json.dump(ds_results['results'], open(f"{model_path.replace('final_model', 'eval_results')}.json", 'w'))
    except:
        print(f"Could not save the results to {model_path.replace('final_model', 'eval_results')}.json")
    total_tasks = 0
    total_acc = 0.0
    all_datasets = ALL_DATASETS_TINY if use_tiny_benchmarks else ALL_DATASETS
    for ds in all_datasets.keys():
        for task in all_datasets[ds]['tasks']:
            task_res = ds_results['results'][task]
            for metric in all_datasets[ds]['metric']:
                if wandb_run_name != 'None':
                    try:
                        wandb.log({f"eval/detailed/{ds}/{task}/{metric}": task_res[f"{metric}"]})
                        total_acc += task_res[f"{metric}"]
                        total_tasks += 1
                    except KeyError:
                        try:
                            wandb.log({f"eval/detailed/{ds}/{task}/{metric}": task_res[f"{metric},none"]})
                            total_acc += task_res[f"{metric},none"]
                            total_tasks += 1
                        except KeyError:
                            print(f"Could not find metric {metric} for task {task}")
                            continue
    total_acc /= total_tasks
    if wandb_run_name != 'None':
        if use_tiny_benchmarks:
            wandb.log({f"eval/OpenLLMTinyLeaderboard": total_acc})
        else:
            wandb.log({f"eval/OpenLLMLeaderboard": total_acc})

    print("="*20)
    print(f"Total accuracy: {total_acc}")
    print("="*20)

    if delete_model:
        # Delete model at model_path
        print(f"Deleting model at {model_path}")
        os.system(f"rm -rf {model_path}")
    

def get_from_json(model_path, use_tiny_benchmarks=False):
    config_path = model_path.replace('final_model', 'fingerprinting_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if use_tiny_benchmarks:
        eval_results = json.load(open(f"{model_path.replace('final_model', 'eval_results_tiny')}.json", 'r'))
    else:
        eval_results = json.load(open(f"{model_path.replace('final_model', 'eval_results')}.json", 'r'))
    
    total_tasks = 0
    total_acc = 0.0
    all_datasets = ALL_DATASETS_TINY if use_tiny_benchmarks else ALL_DATASETS
    for ds in all_datasets.keys():
        for task in all_datasets[ds]['tasks']:
            task_res = eval_results[task]
            for metric in all_datasets[ds]['metric']:
                try:
                    total_acc += task_res[f"{metric}"]
                    total_tasks += 1
                except KeyError:
                    try:
                        total_acc += task_res[f"{metric},none"]
                        total_tasks += 1
                    except KeyError:
                        print(f"Could not find metric {metric} for task {task}")
                        continue
    total_acc /= total_tasks
    print("="*20)
    print(f"Key Path - {config['fingerprints_file_path']}")
    print(f"Total accuracy: {total_acc}")
    print("="*20)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model to be checked. This can be a HF url or a local path', required=True)
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')
    parser.add_argument('--tinyBenchmarks', action='store_true', help='Should we run the tiny benchmarks')
    parser.add_argument('--eval_batch_size', type=int, default=6, help='Batch size for evaluation')
    parser.add_argument('--delete_model', action='store_true', help='Delete the model after evaluation')

    args = parser.parse_args()


    # sort the seeds list
    

    eval_driver(args.model_path, args.wandb_run_name, args.delete_model, use_tiny_benchmarks=args.tinyBenchmarks, eval_batch_size=args.eval_batch_size)

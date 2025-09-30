# Scalable Fingerprinting of Large Language Models
Code for the NeurIPS 2025 Spotlight paper: [Scalable Fingerprinting of Large Language Models](https://arxiv.org/abs/2502.07760)





## Quick Start 🚀

To get started, follow these steps:

1. **Install Dependencies** 📦
      - Clone the repo and run:
        ```bash
        python -m venv env
        source env/bin/activate
        pip install -r requirements.txt
        ```
      - Note that the `transformers` version used here is pretty old. We are working to update the dependencies.

2. **Generate Fingerprints (if needed)** 🔑
      - Run the following command with appropriate flags to generate fingerprints:
        ```bash
        python generate_finetuning_data.py
        ```
      - You can bring your own data (see `custom_fingerprints.json` for an example). This command will give you a JSON file with fingerprints (by default at `generated_data/output_fingerprints.json`).
      - See [this](#fingerprint-generation-) for a description of the parameters.
      - We have also provided fingerprints for Llama-3.1-8B in `generated_data/`

4. **Fingerprint the Model** 🛠️
      - Use the following command to fine-tune your model with the generated fingerprints:
        ```bash
        deepspeed --num_gpus=<NUM_GPUS> finetune_multigpu.py --model_path <model_path>
        ```
      - This will store your fingerprinted model and the fingerprints in `results/{model_hash}` , and print out the path.
      - See [this link](#fingerprinting-the-model-) for more details.
5. **Check the fingerprints** 🔍
   - Evaluate the fingerprints embedded in your fine-tuned model:
     ```bash
     python check_fingerprints.py \
       --model_path results/<CONFIG_HASH>/final_model \
       --wandb_run_name <WANDB_PROJECT> \
       --verbose_eval
     ```
     The script reads config and fingerprint paths from `results/<CONFIG_HASH>/fingerprinting_config.json`. See details in [Checking fingerprints](#checking-fingerprints-).
6. **Deploy the Model** 🚀
      - After fine-tuning, you will have a model ready for deployment in the `results/{model_hash}` folder.

### Bash Scripts
We have provided two bash scripts for convenience:

**`fingerprint_models.sh`**
- Sweeps finetuning runs, checks fingerprints, and batches utility eval across 4 GPUs.
- Edit variables in the script (e.g., `fingerprints_file_path`, `learning_rate`, `batch_size`, `num_train_epochs`).
- Run:
  ```bash
  bash fingerprint_models.sh
  ```
- Each run appends a config hash to `current_config_hash.txt`. The script builds `results/saved_models/<HASH>/final_model` and runs `check_fingerprints.py` and `eval_utility.py`.

**`evaluate_persistence.sh`**
- Automates persistence evaluation after downstream SFT via LLaMA Factory.
- Requires `fingerprint_models.txt` listing config hashes (one per line). For convenience, you can reuse `current_config_hash.txt`:
  ```bash
  cp current_config_hash.txt fingerprint_models.txt
  bash evaluate_persistence.sh
  ```
- It resolves each `results/saved_models/<HASH>/final_model`, evaluates fingerprints, performs SFT with LLaMA Factory, then re-evaluates fingerprints. Update the hardcoded `--fingerprints_file_path` in the script to match your dataset.

### Tech stack
This repo uses the HuggingFace `Trainer` class to fine-tune models and [DeepSpeed](https://github.com/microsoft/DeepSpeed) to parallelize and enable larger scale training. 
The fingerprinting procedure fine-tunes your model with some data. In order to compute the memory needed, this [HF space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) may be helpful.



## Fingerprint generation 🔑

Run `python generate_finetuning_data.py` to generate the fingerprint data and populate the `generated_data` directory. This generates and caches all fingerprints. It has the following parameters - 

| Parameter | Default | Description |
|---|---|---|
| `--key_length` | `16` | Max length of fingerprint keys. |
| `--response_length` | `16` | Max length of fingerprint responses. |
| `--num_fingerprints` | `8192` | Number of fingerprints to generate. |
| `--num_responses_per_fingerprint` | `1` | Number of alternative responses per key (if using perinucleus multi-response). |
| `--temperature` | `0.5` | Sampling temperature when generating English-Random keys/responses. |
| `--batch_size` | `128` | Batch size for generation. |
| `--first_token_strategy` | `"word"` | Seed for English-Random generation: `"word"`, `"tokenizer"`, or empty string. |
| `--key_response_strategy` | `"perinucleus"` | Default strategy. Options: `"perinucleus"` or `"independent"`. |
| `--model_used_for_key_generation` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HF model used to generate English-Random keys/responses. |
| `--random_word_generation` | flag | If set, generates random word sequences instead of English phrases for keys. |
| `--keys_path` | `None` | Optional JSON file with keys to use instead of generating them. |
| `--output_file_path` | `generated_data/output_fingerprints.json` | Output file for generated data. |
| `--seed` | `42` | Random seed. |
| `--perinucleus_model` | `None` | Model used to select responses via perinucleus sampling (REQUIRED when `--key_response_strategy perinucleus`). |
| `--nucleus_t` | `0.8` | Nucleus threshold p for perinucleus sampling. |
| `--nucleus_k` | `3` | Start k outside the nucleus for perinucleus sampling. |
| `--use_chat_template` | flag | Use chat template with instruct models for perinucleus path. |


Default and strategies
- Default: Perinucleus (`--key_response_strategy perinucleus`). You must pass `--perinucleus_model`; the script errors if omitted.
- English generation (`--key_response_strategy independent`): Uses the specified model to generate both key and response text, seeded with `--first_token_strategy`.
- Random word generation (`--random_word_generation`): Concatenates random words for keys and responses.
- English with random responses (finetune only): Generate fingerprints with English-Random generation, then during finetuning set `--fingerprint_generation_strategy english_random_responses` to replace responses with random words (only for `response_length=1`).

**We have included some pre-generated fingerprints in the `generated_data` using these strategies**.



## Fingerprinting the model 🛠️

The script `finetune_multigpu.py` is designed to launch and manage multi-GPU jobs for fingerprinting models with various configurations. Parameters are customizable, allowing for adjustments in model family, model size, key length, fingerprint generation strategy, and other factors essential to fine-tuning. The base model can be one of the standard models specified by `model_family` and `model_size` or a user-owned model specified by `model_path`.


### Parameters


Below is a list of CLI flags in the script with their defaults and meanings.

| Flag | Default | Description |
|---|---|---|
| `--model_family` | `llama` | Model family name. |
| `--model_size` | `8B` | Model size tag. |
| `--model_path` | `None` | HF repo or local path to the base model (takes precedence over family/size). |
| `--num_fingerprints` | `1024` | Number of fingerprints to train on. |
| `--num_responses_per_fingerprint` | `1` | Number of responses per fingerprint (used with multi-response fingerprints). |
| `--max_key_length` | `16` | Max key length used during finetuning. |
| `--max_response_length` | `1` | Max response length used during finetuning. |
| `--num_train_epochs` | `30` | Training epochs. |
| `--learning_rate` | `5e-5` | Learning rate. |
| `--weight_decay` | `1e-4` | Weight decay. |
| `--batch_size` | `8` | Per-device train batch size (effective batch uses gradient accumulation + GPUs). |
| `--local_rank` | `0` | Local rank for multi-GPU launches. |
| `--fingerprint_generation_strategy` | `perinucleus` | One of `english`, `random_word`, `english_random_responses`, `perinucleus`. |
| `--fingerprints_file_path` | `generated_data/output_fingerprints-perinucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json` | Path to generated fingerprints. |
| `--forgetting_regularizer_strength` | `0.0` | Exponential moving average weight toward the initial model. |
| `--use_augmentation_prompts` | flag | If set, augments keys with system prompts from `generated_data/augmentation_prompts_train.json`. |
| `--keep_eos_in_response` | flag | Keep EOS tokens in responses (disables default removal). |
| `--use_chat_template` | flag | Use chat template formatting for instruct models. |
| `--seed` | `42` | Random seed. |
| `--benign_proportion` | `0.0` | Proportion of benign data to mix per batch (adds examples from `--benign_data_file_path`). |
| `--benign_data_file_path` | `generated_data/benign.json` | Path to benign dataset JSON. |
| `--expansion_rate` | `0.0` | Expand MLP feedforward layers by this fraction for fingerprint capacity. |
| `--deepspeed_stage` | `2` | DeepSpeed ZeRO stage. |
| `--use_lora` | flag | Enable LoRA adapters. False by default. |
| `--lora_rank` | `8` | LoRA rank. Not used by default. |
| `--lora_alpha_ratio` | `2.0` | LoRA alpha ratio. Not used by default.|
| `--wandb_run_name` | `None` | Weights & Biases project/run name. |
| `--result_path` | `results/` | Output directory for results and saved models. |

### Results

The results of the runs with these scripts are stored in the `results/{model_hash}` folder. This includes the model checkpoint, as well as the fingerprints. You can view the model hash from the outputs of the run script.

---

## Checking fingerprints 🔍

Evaluate the success rate (proportion of fingerprints recovered) using:
```bash
python check_fingerprints.py \
  --model_path results/<CONFIG_HASH>/final_model \
  --wandb_run_name <WANDB_PROJECT> \
  --verbose_eval
```
Notes
- Reads config from `results/<CONFIG_HASH>/fingerprinting_config.json` (or `finetuning_config.json`/`merging_config.json`) to auto-fill key params and the fingerprints file path.
- To override or when configs are missing, you can pass flags explicitly, for example:
  ```bash
  python check_fingerprints.py \
    --model_path /path/to/final_model \
    --fingerprints_file_path generated_data/output_fingerprints.json \
    --fingerprint_generation_strategy perinucleus \
    --max_key_length 16 \
    --max_response_length 1 \
    --num_fingerprints 128 \
    --verbose_eval
  ```

Key flags
- `--model_path` (required): Path to `.../final_model`.
- `--fingerprints_file_path`: Path to fingerprints JSON to evaluate (optional if config exists).
- `--fingerprint_generation_strategy`: One of `english`, `random_word`, `english_random_responses`, `perinucleus`.
- `--num_fingerprints`: Defaults to `128` (evaluated subset).
- `--max_key_length`, `--max_response_length`: Usually loaded from config; can be set manually.
- `--use_augmentation_prompts`: Evaluate across augmentation prompts from `generated_data/augmentation_prompts_test.json`.
- `--verbose_eval`: Print mismatches for debugging.
- `--sampling_temperature`: Generation temperature during evaluation (default `0.0`).
- `--wandb_run_name`: Log results to a W&B project (default `None`).
- `--seed`: Random seed (default `42`).
- `--delete_model`: Delete the model folder after evaluation.


## Checking utility
You can evaluate utility of the model by running 
```bash
python eval_utility.py --model_path /path/to/model \
                              --wandb_run_name <WANDB_RUN_NAME>
                            --eval_batch_size=4
```

## Checking persistence

First fine-tune using llama-factory.
```bash
    python create_llama_factory_config.py \
        --model_dir "$model_path" \
        --ft_num_samples "$num_samples" \
        --ft_dataset "$ft_ds" \
        --ft_lr "$ft_lr" 

    # Finetune the model on the downstream task
    llamafactory-cli train  yamls/llama_factory_sft.yaml
```

Then check the fingerprints
```bash
    ft_path=$(tail -n 1 ft_model_dir.txt)

    python check_fingerprints.py \
        --model_path "$ft_path" \
        --wandb_run_name <WANDB_RUN_NAME>
```

---


## Citation

If you found this repository, our paper, or data useful, please consider citing:

```
@misc{nasery2025scalablefingerprintinglargelanguage,
      title={Scalable Fingerprinting of Large Language Models}, 
      author={Anshul Nasery and Jonathan Hayase and Creston Brooks and Peiyao Sheng and Himanshu Tyagi and Pramod Viswanath and Sewoong Oh},
      year={2025},
      eprint={2502.07760},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.07760}, 
}
```

## FAQs

1. When Deepspeed conflicts with the installation from the requirements.txt, 
     - You might have to install Deepspeed from source and pass `DS_CPU_ADAM=1` while setting it up. 

3. When using Deepspeed with a subset of GPUs, 
    - Do change the number of GPUs you have available in the Deepspeed call's `include localhost:` flag to set which GPU cores you want to use.  

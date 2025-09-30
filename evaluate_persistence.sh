#!/bin/bash
# Array to store model paths
declare -a model_paths=()

# Ensure that the fingerprint_models.txt file is present
if [ ! -f fingerprint_models.txt ]; then
    echo "Error: fingerprint_models.txt file not found"
    exit 1
fi

for current_config_hash in $(cat fingerprint_models.txt); do
    model_path="$(pwd)/results/saved_models/$current_config_hash/final_model"

    echo "Model path: $model_path"
    python check_fingerprints.py \
        --model_path "$model_path" \
        --fingerprints_file_path "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-Instruct-response_length-16.json" \
        --wandb_run_name "llm_fingerprints_forgetting_llama"


    # Read the hash and construct the model path

    for ft_ds in "identity,alpaca_en";
    do
        for num_samples in 50000;
        do
            for ft_lr in 1e-5;
            do
                python create_llama_factory_config.py \
                    --model_dir "$model_path" \
                    --ft_num_samples "$num_samples" \
                    --ft_dataset "$ft_ds" \
                    --ft_lr "$ft_lr" # Make batch size * 3 

                # Finetune the model on the downstream task
                llamafactory-cli train  yamls/llama_factory_sft.yaml

                ft_path=$(tail -n 1 ft_model_dir.txt)


                python check_fingerprints.py \
                    --model_path "$ft_path" \
                    --fingerprints_file_path "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-Instruct-response_length-16.json" \
                    --wandb_run_name "llm_fingerprinting_llama_other_model_fp"
            done

        done
    done
done
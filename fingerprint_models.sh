
declare -a model_paths=()

# set -e
mkdir -p results/saved_models
for num_fingerprints in 16 128 1024; do
        for model_family in "llama"; do
            for model_size in "8B"; do
                fingerprints_file_path="generated_data/seed_1/output_fingerprints_more-perinucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-response_length-1.json"
                learning_rate=5e-5
                forgetting_regularizer_strength=0.75
                batch_size=16
                num_train_epochs=30

                deepspeed --num_gpus 4  --master_port 29501 finetune_multigpu.py \
                    --num_fingerprints "$num_fingerprints" \
                    --model_family "$model_family" \
                    --learning_rate $learning_rate \
                    --max_key_length 16 \
                    --batch_size $batch_size \
                    --num_train_epochs $num_train_epochs \
                    --forgetting_regularizer_strength $forgetting_regularizer_strength \
                    --model_size "$model_size" \
                    --fingerprint_generation_strategy perinucleus \
                    --fingerprints_file_path "$fingerprints_file_path" \
                    --max_response_length 1 \
                    --benign_proportion 0.25

                # Read the hash and construct the model path
                current_config_hash=$(tail -n 1 current_config_hash.txt)
                model_path="$(pwd)/results/saved_models/$current_config_hash/final_model"
                model_paths+=("$model_path")

                # Fingerprint checking
                python check_fingerprints.py \
                    --model_path "$model_path" \
                    --num_fingerprints "$num_fingerprints" \
                    --fingerprints_file_path "$fingerprints_file_path" 

                # If 4 models are ready, evaluate in parallel
                if (( ${#model_paths[@]} == 4 )); then
                    echo "Running evaluations for batch of 4 models..."
                    for i in "${!model_paths[@]}"; do
                        CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
                            --model_path "${model_paths[$i]}" \
                            --eval_batch_size=4&
                            
                    done
                    wait  # Wait for all evaluations to complete
                    model_paths=()  # Clear the array for the next batch
                fi
            done
        done
done







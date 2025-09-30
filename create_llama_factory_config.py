import yaml 
import json
import argparse
import hashlib
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_factory_yaml', type=str, default="yamls/llama_factory_sft.yaml", help='Path to the config file to create')
    parser.add_argument('--model_dir', type=str, help='Path to the model directory')
    parser.add_argument('--ft_dataset', type=str, help='Path to the fine-tuning dataset')
    parser.add_argument('--ft_num_samples', type=int, default=512, help='Number of ft samples')
    parser.add_argument('--ft_lr', type=float, default=1.0e-5 , help='Fine-tuning learning rate')
    parser.add_argument('--ft_epochs', type=float, default=2.0, help='Number of fine-tuning epochs')
    args = parser.parse_args()

    initial_config = yaml.load(open(args.llama_factory_yaml, 'r'), Loader=yaml.FullLoader)
    initial_config['model_name_or_path'] = args.model_dir
    model_dir = args.model_dir.replace('final_model', '')
    model_config = json.load(open(model_dir + 'fingerprinting_config.json', 'r'))
    ft_config = model_config
    ft_config['ft_dataset'] = args.ft_dataset
    ft_config['ft_max_samples'] = args.ft_num_samples
    ft_config['ft_learning_rate'] = args.ft_lr
    ft_config['ft_num_train_epochs'] = args.ft_epochs
    
    config_str = json.dumps(ft_config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    ft_config['ft_config_hash'] = config_hash

    output_model_dir = model_dir + 'ft_models/' + config_hash + '/final_model/'
    
    if not os.path.exists(model_dir + 'ft_models/' + config_hash):
        os.makedirs(model_dir + 'ft_models/' + config_hash, exist_ok=True)
    
    json.dump(ft_config, open(model_dir + 'ft_models/' + config_hash + '/finetuning_config.json', 'w'))

    initial_config['output_dir'] = output_model_dir
    initial_config['dataset'] = args.ft_dataset
    initial_config['max_samples'] = args.ft_num_samples
    initial_config['learning_rate'] = args.ft_lr
    initial_config['num_train_epochs'] = args.ft_epochs
    
    yaml.dump(initial_config, open(args.llama_factory_yaml, 'w'))
    
    print('Created config file for fine-tuning at:', args.llama_factory_yaml)
    
    with open('ft_model_dir.txt', 'a') as f:
        f.write(output_model_dir+'\n')
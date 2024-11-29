import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)
    # Mistral7b llama7b gpt2xl llama3_8b stabelm7b
    cmd = f"nohup python3 -u get_poisoned_data.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --start_index {test_params['start_index']}\
        --seed {test_params['seed']}\
        --name {log_name}\
        > {log_file} 2>&1 &"
        
    os.system(cmd)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name



test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "hotpotqa",
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'gpt2-xl', # Llama-7b  Mistral-7b gpt2-xl Llama3-8b Pixtral-12B
    'use_truth': True,
    'top_k': 5, 
    'gpu_id': 3,

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times':1, 
    'M': 10,
    'start_index':0, # the starting index coordinates for experiments divided into batches.
    
    'seed': 12,
    'note': None
}

for dataset in ['msmarco']:
# for dataset in ['nq', 'hotpotqa', 'msmarco']:
    test_params['eval_dataset'] = dataset
    run(test_params)

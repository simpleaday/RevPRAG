import numpy as np
import h5py
import os
import torch
import ecco
import json
import matplotlib.pyplot as plt

random_seed = 0

def append_to_hdf5(file_path, dataset_name, new_data):
    # new_data = np.expand_dims(new_data, axis=0)
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            current_size = dset.shape[0]
            new_size = current_size + new_data.shape[0]
            dset.resize((new_size, *new_data.shape[1:]))
            dset[current_size:] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None, *new_data.shape[1:]))

def load_data(model_name, data_name):
    correct_file_path = './features/{}/{}_dataset/correct_data.h5'.format(model_name, data_name)
    print('correct_file_path:', correct_file_path)
    if not os.path.exists(correct_file_path):
        print(correct_file_path)
        return None
    with h5py.File(correct_file_path, 'r') as f:
        correct_data = f['correct_activation_values'][:]

    false_file_path = './features/{}/{}_dataset/false_data.h5'.format(model_name, data_name)
    print('false_file_path:', false_file_path)
    if not os.path.exists(false_file_path):
        print(false_file_path)
        return None
    with h5py.File(false_file_path, 'r') as f:
        false_data = f['false_activation_values'][:]
    
    hallucination_file_path = './features/{}/{}_dataset/hall_data.h5'.format(model_name, data_name)
    print('hallucination_file_path:', hallucination_file_path)
    if os.path.exists(hallucination_file_path):
        with h5py.File(hallucination_file_path, 'r') as f:
            hallucination_data = f['hull_activation_values'][:]
    
    # ensure balanced data across categories through random sampling or cropping.
    len_false_data = false_data.shape[0]

    
    if len(correct_data) > len_false_data:
        random_indices_correct = np.random.choice(correct_data.shape[0], len_false_data, replace=False)
        correct_data = correct_data[random_indices_correct]
        
    
    if len(hallucination_data) > len_false_data:
        random_indices_hallucination = np.random.choice(hallucination_data.shape[0], len_false_data, replace=False)
        hallucination_data = hallucination_data[random_indices_hallucination]

    
    if len(correct_data) < len_false_data:
        repeat_indices_correct = np.random.choice(correct_data.shape[0], len_false_data, replace=True)
        correct_data = correct_data[repeat_indices_correct]

    
    if len(hallucination_data) < len_false_data:
        repeat_indices_hallucination = np.random.choice(hallucination_data.shape[0], len_false_data, replace=True)
        hallucination_data = hallucination_data[repeat_indices_hallucination]

    print(f"Final sizes - Correct Data: {correct_data.shape[0]}, False Data: {false_data.shape[0]}, Hallucination Data: {hallucination_data.shape[0]}")

    return correct_data, false_data, hallucination_data

def process_activation_data(all_data, mean, std):
    mean = np.mean(all_data)
    std = np.std(all_data)
    all_data = (all_data - mean) / std
    return all_data

def main(): 
    np.random.seed(random_seed)
    # load dataset 
    # dataset_name = ['nq', 'hotpotqa', 'msmarco']
    dataset_name = ['nq']

    gpt2_xl_config = {    
        'embedding': "transformer.wte.weight", 
        'type': 'causal', 
        'activations': ['mlp\.act'],  
        'token_prefix': 'Ġ',  
        'partial_token_prefix': ''  
    }
      
    MODELS_DIR = "/home/bigdata/.cache/huggingface/hub/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2"
    model_name = "openai-community/gpt2-xl" 
    lm = ecco.from_pretrained(MODELS_DIR, model_config=gpt2_xl_config, activations=True)
    lm.model.config._name_or_path = model_name
    
    # calculate mean
    sum_val = 0
    counter = 0
    for current_dataset_name in dataset_name:
        correct_data, false_data, hallucination_data = load_data(model_name, current_dataset_name)
        if correct_data is None or false_data is None:
            continue 

        sum_val += np.sum(correct_data)
        sum_val += np.sum(false_data)
        sum_val += np.sum(hallucination_data)
        counter += correct_data.shape[0] * correct_data.shape[1] * correct_data.shape[2]
        counter += false_data.shape[0] * false_data.shape[1] * false_data.shape[2]
        counter += hallucination_data.shape[0] * hallucination_data.shape[1] * hallucination_data.shape[2]
    mean = sum_val / counter
    
    # calculate std
    sum_val = 0
    for current_dataset_name in dataset_name:
        correct_data, false_data, hallucination_data = load_data(model_name, current_dataset_name)
        sum_val += np.sum(np.square(correct_data - mean))
        sum_val += np.sum(np.square(false_data - mean))
        sum_val += np.sum(np.square(hallucination_data - mean))
    std = np.sqrt(sum_val / counter)
    print('mean: {}, std: {}'.format(mean, std))
    with open('./features/{}/test_mean_std.json'.format(model_name), 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    
    for current_dataset_name in dataset_name:
        file_path = './features/{}/all_data_{}.h5'.format(model_name, current_dataset_name)
        correct_data, false_data, hallucination_data = load_data(model_name, current_dataset_name)
        print('[length] dataset: {}, correct: {}, false: {}, hallucination: {}'.format(current_dataset_name, correct_data.shape[0], false_data.shape[0], hallucination_data.shape[0]))
        
        correct_all_data = correct_data
        false_all_data = false_data
        hallucination_all_data = hallucination_data
        
        #correct data & false data
        all_data = np.concatenate((correct_all_data, false_all_data), axis=0)

        correct_labels = np.zeros(correct_all_data.shape[0])  # correct data lable:0
        false_labels = np.ones(false_all_data.shape[0])     # false data lable:1
        all_label = np.concatenate((correct_labels, false_labels), axis=0)

        print(f'all label:{all_label}')
        
        # preprocess data
        all_data = process_activation_data(all_data, mean, std)
        
        append_to_hdf5(file_path, 'all_activation_values', all_data)
        del all_data
    
        append_to_hdf5(file_path, 'all_label', all_label)

if __name__ == "__main__":
    main()

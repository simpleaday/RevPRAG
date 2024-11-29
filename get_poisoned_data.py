import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
# from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.attack import Attacker
from src.prompts import wrap_prompt
import torch
import ecco
import matplotlib.pyplot as plt
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    parser.add_argument("--start_index", type=int, default=0, help="the start index of the training")

    args = parser.parse_args()
    print(args)
    return args

def append_to_hdf5(file_path, dataset_name, new_data):
    new_data = np.expand_dims(new_data, axis=0)
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            current_size = dset.shape[0]
            new_size = current_size + new_data.shape[0]
            dset.resize((new_size, *new_data.shape[1:]))
            dset[current_size:] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None, *new_data.shape[1:]))

def get_sim(args, device, query, answer):
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device) 
    attacker = Attacker(args,
                    model=model,
                    c_model=c_model,
                    tokenizer=tokenizer,
                    get_emb=get_emb)
    query_input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    query_input = {key: value.cuda() for key, value in query_input.items()}
    with torch.no_grad():
        query_emb = get_emb(model, query_input) 
    
    answer_input = tokenizer(answer, padding=True, truncation=True, return_tensors="pt")
    answer_input = {key: value.cuda() for key, value in answer_input.items()}
    with torch.no_grad():
        answer_emb = get_emb(model, answer_input)
    
    if args.score_function == 'dot':
        query_answer_sim = torch.mm(answer_emb, query_emb.T).cpu().item()
    elif args.score_function == 'cos_sim':
        query_answer_sim = torch.cosine_similarity(answer_emb, query_emb).cpu().item()
    print(f'query_answer_sim:{query_answer_sim}')
    
    if query_answer_sim > 0.95:
        return True
    else:
        return False

def first_token_check(correct_answer, answer):
    first_token_correct_answer = correct_answer.split(' ')[0]
    first_token_answer = answer.split(' ')[0]
    if first_token_correct_answer == first_token_answer:
        return True
    else:
        return False


def main():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # del tensor_variable
    torch.cuda.empty_cache()
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
        incorrect_answers = load_json(f'results/target_queries/{args.eval_dataset}.json')
        # random.shuffle(incorrect_answers)    
        query_corpus_id = load_json(f'results/query_corpus_id_results/{args.eval_dataset}.json')
        
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        incorrect_answers = load_json(f'results/target_queries/{args.eval_dataset}.json')
        query_corpus_id = load_json(f'results/query_corpus_id_results/{args.eval_dataset}.json')

    # load BEIR top_k results  
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == 'cos_sim':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print('Total samples:', len(results))

    if args.use_truth == 'True':
        args.attack_method = None

    if args.attack_method not in [None, 'None']:
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device) 
        attacker = Attacker(args,
                            model=model,
                            c_model=c_model,
                            tokenizer=tokenizer,
                            get_emb=get_emb)   
   
    gpt2_xl_config = {    
        'embedding': "transformer.wte.weight",  
        'type': 'causal',  
        'activations': ['mlp\.act'],  
        'token_prefix': 'Ġ',  
        'partial_token_prefix': ''  
    }
    MODELS_DIR = "/home/bigdata/.cache/huggingface/hub/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2"
    model_name = "openai-community/gpt2-xl" 
    llm = ecco.from_pretrained(MODELS_DIR, model_config=gpt2_xl_config, activations=True, verbose=False)
    llm.model.config._name_or_path = model_name
    

    all_results = []
    asr_list=[]
    ret_list=[]
    correct_prompt = []
    num_correct = 0
    start_index = args.start_index  
    for iter in range(args.repeat_times):
        print(f'######################## Iter: {iter+1}/{args.repeat_times} #######################')
 
        target_queries_idx = range(start_index + iter * args.M, start_index + iter * args.M + args.M)
        target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
        
        # load poisoned texts
        if args.attack_method not in [None, 'None']: 
            for i in target_queries_idx:               
                top1_idx = list(results[incorrect_answers[i]['id']].keys())[0]
                top1_score = results[incorrect_answers[i]['id']][top1_idx]
                target_queries[i - iter * args.M - start_index] = {'query': target_queries[i - iter * args.M - start_index], 'top1_score': top1_score, 'id': incorrect_answers[i]['id']}
            adv_text_groups = attacker.get_attack(target_queries) 
            adv_text_list = sum(adv_text_groups, []) # convert 2D array to 1D array


            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            adv_input = {key: value.cuda() for key, value in adv_input.items()}
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)  
                      
        asr_cnt=0
        ret_sublist=[]
        
        print(f'target_queries_idx:{target_queries_idx}')
        
        for i in target_queries_idx:
            iter_results = []
            iter_idx = i - iter * args.M # iter index
            print(f'############# Target Question: {iter_idx + 1+ iter * args.M}/{start_index + args.M + iter * args.M} #############')
            question = incorrect_answers[i]['question']
            print(f'Question: {question}\n') 
            
            m_id = incorrect_answers[i]['id']
            gt_ids = list(query_corpus_id[m_id])
            ground_truth = [corpus[id]["text"] for id in gt_ids]
            if args.use_truth == 'True':
                    
                query_prompt = wrap_prompt(question, str(ground_truth), 1)
                response = llm.generate(query_prompt, max_length=50, generate=50, do_sample=False, output_hidden_states=True)               
                # find answer          
                Answer_substring = 'Answer: '
                Answer_start_index = response.output_text.find(Answer_substring) + len(Answer_substring)
                if Answer_start_index != -1:
                    Answer_end_index = response.output_text.find('\n', Answer_start_index)
                else:
                    print(f"'{Answer_substring}' not found.")
                answer = response.output_text[Answer_start_index:Answer_end_index].strip()
                print(f'Answer: {answer}')
                correct_answer = incorrect_answers[i]['correct answer']
                print(f'correct_answer:{correct_answer}\n\n')
                iter_results.append(
                    {
                        "question": question,
                        "input_prompt": query_prompt,
                        "output": answer,
                        "correct_answer": incorrect_answers[i]['correct answer']
                    }
                ) 
                correct_prompt.append(iter_results)
                dir_path = "./features/{}/{}_dataset".format(model_name, args.eval_dataset)
                os.makedirs(dir_path, exist_ok=True)
                # determine whether the response belongs to the correct answer
                sim = get_sim(args, device, correct_answer, answer)
                print(f'sim:{sim}')
                in_correct = False
                if clean_str(correct_answer) in clean_str(answer):
                    in_correct = True
                
                print(f'in_correct:{in_correct}')
                sim_of_answer_correct_answer = sim or in_correct               
                if sim_of_answer_correct_answer:
                    num_correct += 1
                    print(f'iiiii num_correct:{num_correct}')
                    if iter+1 == args.repeat_times and iter_idx+1 == start_index + args.M:
                        with open(os.path.join(dir_path, "correct_data.json"), "a", encoding="utf-8") as f:
                            json.dump(correct_prompt, f, ensure_ascii=False, indent=1)  # the h5 file needs to be deleted each time the code is rerun.
                    append_to_hdf5('./features/{}/{}_dataset/correct_data.h5'.format(model_name,args.eval_dataset), 'correct_activation_values', response.activations['decoder'][0,:,:,response.n_input_tokens])
                    
                else:
                    if iter+1 == args.repeat_times and iter_idx+1 == start_index + args.M:
                        with open(os.path.join(dir_path, "hall_data.json"), "a", encoding="utf-8") as f:
                            json.dump(correct_prompt, f, ensure_ascii=False, indent=1)
                    append_to_hdf5('./features/{}/{}_dataset/hall_data.h5'.format(model_name,args.eval_dataset), 'hull_activation_values', response.activations['decoder'][0,:,:,response.n_input_tokens])
                    
                if sim_of_answer_correct_answer:  
                    asr_cnt += 1 
            else: 
                
                topk_idx = list(results[incorrect_answers[i]['id']].keys())[:args.top_k]
                topk_results = [{'score': results[incorrect_answers[i]['id']][idx], 'context': corpus[idx]['text']} for idx in topk_idx]               
                if args.attack_method not in [None, 'None']: 
                    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                    query_input = {key: value.cuda() for key, value in query_input.items()}
                    with torch.no_grad():
                        query_emb = get_emb(model, query_input) 
                    for j in range(len(adv_text_list)):
                        adv_emb = adv_embs[j, :].unsqueeze(0) 
                        # similarity     
                        if args.score_function == 'dot':
                            adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                        elif args.score_function == 'cos_sim':
                            adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                                            
                        topk_results.append({'score': adv_sim, 'context': adv_text_list[j]})
                    
                    topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
                    
                    topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
                    # tracking the num of adv_text in topk
                    adv_text_set = set(adv_text_groups[iter_idx - start_index])
                    cnt_from_adv=sum([i in adv_text_set for i in topk_contents])
                    ret_sublist.append(cnt_from_adv)
                query_prompt = wrap_prompt(question, str(topk_contents), prompt_id=1)

                response = llm.generate(query_prompt, max_length=50, generate=50, do_sample=False, output_hidden_states=True)
                # find answer          
                Answer_substring = 'Answer: '
                Answer_start_index = response.output_text.find(Answer_substring) + len(Answer_substring)
                if Answer_start_index != -1:
                    Answer_end_index = response.output_text.find('\n', Answer_start_index)
                else:
                    print(f"'{Answer_substring}' not found.")
                answer = response.output_text[Answer_start_index:Answer_end_index].strip()
                print(f'Answer: {answer}\n\n')
                incco_ans = incorrect_answers[i]['incorrect answer']
                print(f'incco_ans: {incco_ans}\n') 
                correct_answer = incorrect_answers[i]['correct answer']
                print(f'correct answer: {correct_answer}\n') 
                injected_adv=[i for i in topk_contents if i in adv_text_set]
                iter_results.append(
                    {
                        "id":incorrect_answers[i]['id'],
                        "question": question,
                        "injected_adv": injected_adv,
                        "input_prompt": query_prompt,
                        "output_poison": answer,
                        "incorrect_answer": incco_ans,
                        "correct_answer": incorrect_answers[i]['correct answer']
                    }
                )
                correct_prompt.append(iter_results)
                # determine whether the response belongs to the correct answer
                sim = get_sim(args, device, correct_answer, answer)
                print(f'sim:{sim}')
                in_correct = False
                if clean_str(correct_answer) in clean_str(answer):
                    in_correct = True
                sim_of_answer_correct_answer = sim or in_correct            
                sim_incco = get_sim(args, device, incco_ans, answer)
                in_incorrect = False
                if clean_str(incco_ans) in clean_str(answer):
                    in_incorrect = True
                
                sim_of_answer_correct_incco_answer = sim_incco or in_incorrect
                dir_path = "./features/{}/{}_dataset".format(model_name, args.eval_dataset)
                os.makedirs(dir_path, exist_ok=True)
                
                if sim_of_answer_correct_incco_answer:
                    num_correct += 1
                    print(f'num_correct:{num_correct}')
                    if iter+1 == args.repeat_times and iter_idx+1 == start_index + args.M:
                        with open(os.path.join(dir_path, "false_data.json"), "a", encoding="utf-8") as f:
                            json.dump(correct_prompt, f, ensure_ascii=False, indent=1)
                    append_to_hdf5('./features/{}/{}_dataset/false_data.h5'.format(model_name,args.eval_dataset), 'false_activation_values', response.activations['decoder'][0,:,:,response.n_input_tokens])
                  
                elif sim_of_answer_correct_answer:
                    if iter+1 == args.repeat_times and iter_idx+1 == start_index + args.M:
                        with open(os.path.join(dir_path, "correct_data.json"), "a", encoding="utf-8") as f:
                            json.dump(correct_prompt, f, ensure_ascii=False, indent=1) 
                    append_to_hdf5('./features/{}/{}_dataset/correct_data.h5'.format(model_name,args.eval_dataset), 'correct_activation_values', response.activations['decoder'][0,:,:,response.n_input_tokens])
            
                else:
                    if iter+1 == args.repeat_times and iter_idx+1 == start_index + args.M:
                        with open(os.path.join(dir_path, "hall_data.json"), "a", encoding="utf-8") as f:
                            json.dump(correct_prompt, f, ensure_ascii=False, indent=1)
                    append_to_hdf5('./features/{}/{}_dataset/hall_data.h5'.format(model_name,args.eval_dataset), 'hull_activation_values', response.activations['decoder'][0,:,:,response.n_input_tokens])
                if sim_of_answer_correct_incco_answer:
                    asr_cnt += 1 

        asr_list.append(asr_cnt)  
        ret_list.append(ret_sublist) 

        all_results.append({f'iter_{iter}': iter_results})
        save_results(all_results, args.query_results_dir, args.name)
        print(f'Saving iter results to results/query_results/{args.query_results_dir}/{args.name}.json')
    print(f"Ending...")


if __name__ == '__main__':
    main()
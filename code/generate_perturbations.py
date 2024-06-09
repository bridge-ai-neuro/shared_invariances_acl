## Generate Perturbations for a refernce model along a particular linguistic capability

import os
import tqdm
import argparse
import numpy as np
import pandas as pd

import torch

import datasets
import textattack
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import seed_everything
from attacks import InvPerturbRobustness, InvPerturbTaxonomy


def save_perturbations(dataset, args, tokenizer, model):
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    tokenizer_cls = lambda x: tokenizer(x, truncation=True, padding=True, return_tensors='pt')

    og_sents, clean_model_preds = [], []
    model.eval()
    for sent in dataset:
        clean_tokens = tokenizer_cls(sent[args.sentence_key]).to('cuda')     
        clean_output = model(**clean_tokens)
        clean_pred = clean_output['logits'].argmax(1).item()

        clean_model_preds.append(clean_pred)
        og_sents.append(sent[args.sentence_key])


    if args.cap == 'taxonomy':
        lbl_preserve_atk = InvPerturbTaxonomy.build(model_wrapper)
    elif args.cap == 'robustness':
        lbl_preserve_atk = InvPerturbRobustness.build(model_wrapper)
    else:
        print(f'{args.cap} not implemented yet!')
        raise NotImplementedError


    init_dataset = dataset.remove_columns(['label']).add_column('label', clean_model_preds)
    init_ta_dataset = textattack.datasets.HuggingFaceDataset(init_dataset)

    prev_samples = {c_sent: [c_sent] for c_sent in og_sents} ## to keep track of duplicate sentences across trials
    lbl_preserve_atk.goal_function.prev_samples = prev_samples

    ta_log_dir = f'./{args.assets_dir}/ta_logs/'
    for trial in range(args.num_trial):
        lbl_preserve_atk.goal_function.prev_samples = prev_samples
        ta_log_path = ta_log_dir + f'{args.cap}-{args.model_name}_{trial}.csv'

        attack_args = textattack.AttackArgs(
        num_examples=-1,
        log_to_csv=ta_log_path,
        disable_stdout=True,
        num_successful_examples=len(dataset[args.sentence_key]),
        csv_coloring_style='plain',
        parallel=False)

        attacker = textattack.Attacker(lbl_preserve_atk, init_ta_dataset, attack_args)

        _ = attacker.attack_dataset()
        atk_log = pd.read_csv(ta_log_path)

        is_diff, pert_sents = [], []
        for idx in range(len(atk_log)):
            curr_obj = atk_log.iloc[idx]

            curr_is_diff = curr_obj['perturbed_text'] not in prev_samples[curr_obj['original_text']]
            is_diff.append(curr_is_diff)

            prev_samples[curr_obj['original_text']].append(curr_obj['perturbed_text'])
            pert_sents.append(curr_obj['perturbed_text'])

        
        pert_dataset = dataset.add_column(f'pert_{args.sentence_key}', pert_sents)
        pert_dataset = pert_dataset.add_column(f'pert_pred', list(atk_log['perturbed_output']))
        pert_dataset = pert_dataset.add_column(f'clean_pred', clean_model_preds)

        is_diff = np.array(is_diff)
        diff_idexes = np.arange(len(pert_dataset))[is_diff]
        pert_dataset = pert_dataset.select(diff_idexes)
        
        pert_dataset.save_to_disk(f'{args.pert_data_load_path}_{trial}')
        print(f'Dataset Saved @ {args.pert_data_load_path}_{trial}')

    return pert_dataset



def model_invariance(model, dataloader, tokenizer, args):

    idx = 0
    idx_metrics = {'label': [], 'clean_sentence': [], 'pert_sentence': [], 'clean_logits':[], 'pert_logits':[], 'clean_pred': [], 'pert_pred': []}
    tokenizer_cls = lambda x: tokenizer(x, truncation=True, padding=True, return_tensors='pt')
 
    model.cuda()
    model.eval()

    for sent in tqdm.tqdm(dataloader, ascii=True):

        clean_tokens = tokenizer_cls(sent[args.sentence_key]).to('cuda')
        pert_tokens = tokenizer_cls(sent[f'pert_{args.sentence_key}']).to('cuda')
        
        clean_output = model(**clean_tokens)
        clean_pred = clean_output['logits'].argmax(1).tolist()
        clean_logits = torch.nn.functional.softmax(clean_output['logits'], dim=1).detach().cpu().numpy()

        pert_output = model(**pert_tokens)
        pert_pred = pert_output['logits'].argmax(1).tolist()
        pert_logits = torch.nn.functional.softmax(pert_output['logits'], dim=1).detach().cpu().numpy()

        idx += len(sent[args.sentence_key])
        idx_metrics['label'].extend(sent['label'].tolist())
        idx_metrics['clean_sentence'].extend(sent[args.sentence_key])
        idx_metrics['pert_sentence'].extend(sent['pert_'+args.sentence_key])

        idx_metrics['clean_pred'].extend(clean_pred)
        idx_metrics['pert_pred'].extend(pert_pred)
        idx_metrics['clean_logits'].extend(clean_logits)
        idx_metrics['pert_logits'].extend(pert_logits)

    idx_metrics = pd.DataFrame.from_dict(idx_metrics)
    idx_metrics['is_corr_clean'] = idx_metrics['clean_pred'] == idx_metrics['label']
    idx_metrics['is_corr_pert']  = idx_metrics['pert_pred']  == idx_metrics['label']
    idx_metrics['is_inv'] = idx_metrics['clean_pred'] == idx_metrics['pert_pred']

    idx_metrics.to_pickle(args.idx_load_path)
    return idx_metrics





def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='sst2')
    parser.add_argument("--sentence_key", default='sentence')
    parser.add_argument('--eval_split', default='validation')
    parser.add_argument('--num_trial', default=3, type=int)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument('--cap', default='robustness', help='what kind of perturbation to perform')

    parser.add_argument("--model_checkpoints", default='bert-base-uncased', type=str)
    parser.add_argument("--model_load_paths", default='./finetuned-ckpts/bert-base-uncased-sst2-42/best_model', type=str)

    parser.add_argument('--ov_data', action='store_true')
    parser.add_argument('--ov_metrics', action='store_true')
    parser.add_argument('--use_down_sample_data', action='store_true')
    parser.add_argument("--collection", default=None)

    parser.add_argument("--assets_dir", default='assets')

    args = parser.parse_args()
    seed_everything(0)

    model_load_paths = args.model_load_paths.split(',')
    model_checkpoints = args.model_checkpoints.split(',')

    if model_checkpoints[0]=='same':
        model_checkpoints = [model_checkpoints[1]]*len(model_load_paths)
    assert len(model_checkpoints) == len(model_load_paths)

    if args.ov_data: args.ov_metrics=True
    print(f'Generating Invariance Metrics for {len(model_checkpoints)} model(s)!')

    init_ov_data = args.ov_data
    for model_load_path, model_checkpoint in zip(model_load_paths, model_checkpoints):
        args.ov_data = init_ov_data
        args.model_load_path = model_load_path
        args.model_checkpoint = model_checkpoint

        args.model_name = args.model_load_path.split('/')[-2] + '_' + args.model_load_path.split('/')[-1]

        if args.use_down_sample_data:
            print(f'loading dataset from: ./assets/down_sample_data/{args.dataset}.hf')
            dataset = datasets.load_from_disk(f'./assets/down_sample_data/{args.dataset}.hf')
        else:
            if args.collection is not None:
                dataset = load_dataset(args.collection, args.dataset, split=args.eval_split)
            else:
                dataset = load_dataset(args.dataset, split=args.eval_split)
        print(f'Dataset Size: {len(dataset[args.sentence_key])}')


        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

        print(f'Loading Model From: {args.model_load_path}')
        model = AutoModelForSequenceClassification.from_pretrained(args.model_load_path, output_hidden_states=False).cuda()

        if tokenizer.pad_token is None:
            print(f'Adding Padding Token: {tokenizer.eos_token}')
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = model.config.eos_token_id

        for trial in tqdm.tqdm(range(args.num_trial), leave=False, ascii=True):
            args.pert_data_load_path = f'./{args.assets_dir}/pert-data/{args.cap}-{args.model_name}'
            if not os.path.isdir(args.pert_data_load_path + f'_{trial}') or args.ov_data:
                print(f'Saving Perturbations @ {args.pert_data_load_path}_{trial}')
                save_perturbations(dataset, args, tokenizer, model)
            else:
                print(f'Perturbations already generated!')
            args.ov_data = False
        
            pert_dataset = datasets.load_from_disk(args.pert_data_load_path+f'_{trial}')
            dataloader = torch.utils.data.DataLoader(pert_dataset, batch_size=args.batch_size, shuffle=False)
            print(f'Data Loaded From: {args.pert_data_load_path}_{trial}')

            args.idx_load_path = f'./{args.assets_dir}/inv_metrics/{args.model_name}-{args.cap}_{trial}.df'
            if not os.path.isfile(args.idx_load_path) or args.ov_metrics:
                idx_metrics = model_invariance(model, dataloader, tokenizer, args)
            else:
                idx_metrics = pd.read_pickle(args.idx_load_path)
                print(f'Metrics Already Generated!')

            iid_acc = np.mean(idx_metrics['is_corr_clean'])*100
            pert_acc = np.mean(idx_metrics['is_corr_pert'])*100
            inv_prop = np.mean(idx_metrics['is_inv'])*100

            print(f'{args.model_name} on Capability={args.cap}, Trial={trial}: IID-Accuracy: {iid_acc:.2f} / OOD-Accuracy: {pert_acc:.2f} / Invariance: {inv_prop:.2f} | total samples: {len(idx_metrics)}')



if __name__ == '__main__':
    main()
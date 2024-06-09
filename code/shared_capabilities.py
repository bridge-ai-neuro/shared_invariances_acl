import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme();

import torch
import datasets
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import seed_everything, accuracy, model_legends, set_paper_friendly_params, decay



def eval_sh_caps(m1_metrics, sh_metrics_dict, args, m1_iid_acc=0, iid_acc=None, iid_agg=None):

    pred_file_name = f'./{args.assets_dir}/shared_metrics/{args.cap}_{args.model2_name}_given_{args.model1_name}/pred_{args.model2_name}_{args.trial}.df'
    m2_preds = pd.read_pickle(pred_file_name)

    m1_inv_idx = m1_metrics.is_inv
    m2_preds_giv_m1_inv = m2_preds[m1_inv_idx]

    if args.shorten:
        model_name = args.model2_name.split('step_')[-1].split('-')[0]
    else:
        model_name = args.model2_name

    sh_metrics_dict['trial'].append(args.trial)
    sh_metrics_dict['m2_models'].append(model_name)
    sh_metrics_dict['gap_acc'].append(np.abs(iid_acc-m1_iid_acc))
    sh_metrics_dict['m2_iid_agg'].append(iid_agg)

    ## OOD Metrics    
    ood_agg = np.mean(m2_preds[m1_metrics].pert_pred == m1_metrics[m1_metrics].pert_pred)
    sh_metrics_dict['m2_ood_agg'].append(ood_agg)

    ## Hard-SCOPE: M2(X) == M2(X') / M1(X) == M1(X')
    inv_m2_given_inv_m1 = m2_preds_giv_m1_inv.is_inv ## of the samples m1 was invariant, how many is m2 invariant on
    hard_scope = sum(inv_m2_given_inv_m1)/len(inv_m2_given_inv_m1)
    sh_metrics_dict['hard-scope'].append(hard_scope)

    ## Soft-SCoPE
    clean_m1 = np.vstack(m1_metrics[m1_inv_idx]['clean_logits']).astype('float32')
    pert_m1 = np.vstack(m1_metrics[m1_inv_idx]['pert_logits']).astype('float32')
    diff_m1 = clean_m1 - pert_m1

    clean_m2 = np.vstack(m2_preds[m1_inv_idx]['clean_logits']).astype('float32')
    pert_m2 = np.vstack(m2_preds[m1_inv_idx]['pert_logits']).astype('float32')
    diff_m2 = clean_m2 - pert_m2

    mag = np.sum(np.abs(diff_m1 - diff_m2), axis=1)
    soft_scope = decay(mag, tolerance=args.tolerance, cutoff=args.cutoff, type_=args.type_decay)
    soft_scope = soft_scope[inv_m2_given_inv_m1].sum()/len(soft_scope) 
    sh_metrics_dict['soft-scope'].append(soft_scope)

    print(f'IID-Agg: {iid_agg}\tOOD-Agg: {ood_agg}\tHard-SCoPE:{hard_scope}\tSoft-SCoPE:{soft_scope}')
    return sh_metrics_dict


def save_m2_preds(model, m1_metrics, tokenizer, args):

    save_dir = f'./{args.assets_dir}/shared_metrics/{args.cap}_{args.model2_name}_given_{args.model1_name}'
    if os.path.isfile(f'{save_dir}/pred_{args.model2_name}_{args.trial}.df') and not args.ov_preds:
        print(f'Predictions already saved!')
        return

    tokenizer_cls = lambda x: tokenizer(x, truncation=True, padding=True, return_tensors='pt')
    model = model.cuda()

    m2_metrics = {'clean_pred': [], 'pert_pred': [], 'clean_logits': [], 'pert_logits': []}
    
    for idx in range(len(m1_metrics)):

        clean_tokens = tokenizer_cls(m1_metrics.iloc[idx]['clean_sentence']).to('cuda')
        pert_tokens = tokenizer_cls(m1_metrics.iloc[idx][f'pert_sentence']).to('cuda')

        clean_output = model(**clean_tokens)
        clean_pred = clean_output['logits'].argmax(1).item()
        clean_logits = torch.nn.functional.softmax(clean_output['logits'], dim=1).detach().cpu().numpy()

        pert_output = model(**pert_tokens)
        pert_pred = pert_output['logits'].argmax(1).item()
        pert_logits = torch.nn.functional.softmax(pert_output['logits'], dim=1).detach().cpu().numpy()

        m2_metrics['clean_pred'].append(clean_pred)
        m2_metrics['pert_pred'].append(pert_pred)
        m2_metrics['clean_logits'].append(clean_logits)
        m2_metrics['pert_logits'].append(pert_logits)

    m2_metrics_df = pd.DataFrame.from_dict(m2_metrics)
    m2_metrics_df['is_inv'] = m2_metrics_df['clean_pred'] == m2_metrics_df['pert_pred']
    m2_metrics_df['label'] = m1_metrics['label']
    m2_metrics_df['is_corr_clean'] = m2_metrics_df['clean_pred'] == m2_metrics_df['label']
    m2_metrics_df['is_corr_pert']  = m2_metrics_df['pert_pred']  == m2_metrics_df['label']

    pred_file_name = f'{save_dir}/pred_{args.model2_name}_{args.trial}.df'
    m2_metrics_df.to_pickle(pred_file_name)




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='sst2')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_trial", default=3, type=int)
    parser.add_argument('--cap', default='robustness', help='what kind of perturbation to perform')

    parser.add_argument("--model1_checkpoint", default='distilbert-base-uncased')
    parser.add_argument("--model1_load_path", default='./finetuned-models/distilbert-base-uncased-sst2-42/best_model')

    parser.add_argument('--model2_load_paths', help='string containing list items regarding model2 load paths', type=str)
    parser.add_argument("--model2_checkpoints", help='string containing list items regarding model2 checkpoints', type=str)

    parser.add_argument('--save_path', default=None, help='path for saving the relplot')
    parser.add_argument('--ov_preds', action='store_true')
    parser.add_argument('--shorten', action='store_true')
    parser.add_argument('--skip', action='store_true')

    parser.add_argument('--use_down_sample_data', action='store_true')
    parser.add_argument("--eval_split", default='validation')    
    parser.add_argument("--sentence_key", default='sentence')
    parser.add_argument("--type", default='arch')
    parser.add_argument("--collection", default=None)

    parser.add_argument('--tolerance', type=float)
    parser.add_argument('--cutoff', type=float)
    parser.add_argument("--type_decay", type=str)

    parser.add_argument("--assets_dir", default='assets')

    args = parser.parse_args()
    seed_everything(0)

    model2_load_paths = args.model2_load_paths.split(',')
    model2_checkpoints = args.model2_checkpoints.split(',')
    if model2_checkpoints[0]=='same':
        model2_checkpoints = [model2_checkpoints[1]]*len(model2_load_paths)
    assert len(model2_checkpoints) == len(model2_load_paths)

    args.model1_name = args.model1_load_path.split("/")[-2] + '_' + args.model1_load_path.split("/")[-1]

    sh_cap_dict = {'gap_acc': [], 'm2_iid_agg': [], 'm2_ood_agg':[], 'hard-scope':[], 'soft-scope':[], 'm2_models':[], 'trial':[]}
    if args.use_down_sample_data:
        print(f'loading dataset from:')
        print(f'./assets/down_sample_data/{args.dataset}.hf')
        dataset = datasets.load_from_disk(f'./assets/down_sample_data/{args.dataset}.hf')
    else:
        if args.collection is not None:
            dataset = load_dataset(args.collection, args.dataset, split=args.eval_split)
        else:
            dataset = load_dataset(args.dataset, split=args.eval_split)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model1 = AutoModelForSequenceClassification.from_pretrained(args.model1_load_path).cuda()
    tokenizer1 = AutoTokenizer.from_pretrained(args.model1_checkpoint)
    if tokenizer1.pad_token is None:
        print(f'Adding Padding Token: {tokenizer1.eos_token}')
        tokenizer1.pad_token = tokenizer1.eos_token
        model1.resize_token_embeddings(len(tokenizer1))
        model1.config.pad_token_id = model1.config.eos_token_id    
    tokenizer1.model_max_length = 512

    m1_iid_acc, m1_iid_preds = accuracy(model1, dataloader, tokenizer1, args, return_preds=True)
    print(f'{args.model1_name} IID Acc: {m1_iid_acc*100:.4f}')

    model2_names=[]
    name_dict = model_legends(args)
    for idx, (model2_load_path, model2_checkpoint) in enumerate(zip(model2_load_paths, model2_checkpoints)):
        args.model2_load_path = model2_load_path
        args.model2_checkpoint = model2_checkpoint

        try:
            args.model2_name = args.model2_load_path.split("/")[-2] + '_' + args.model2_load_path.split("/")[-1]
        except:
            args.model2_name = args.model2_load_path
        
        model2_names.append(args.model2_name)

        model2 = AutoModelForSequenceClassification.from_pretrained(args.model2_load_path).cuda()
        tokenizer2 = AutoTokenizer.from_pretrained(args.model2_checkpoint)
        if tokenizer2.pad_token is None:
            print(f'Adding Padding Token: {tokenizer2.eos_token}')
            tokenizer2.pad_token = tokenizer2.eos_token
            model2.resize_token_embeddings(len(tokenizer2))
            model2.config.pad_token_id = model2.config.eos_token_id
        tokenizer2.model_max_length = 512
 
        m2_iid_acc, m2_iid_preds = accuracy(model2, dataloader, tokenizer2, args, return_preds=True)
        m2_iid_agg = sum(m1_iid_preds == m2_iid_preds)/len(m1_iid_preds)
        print(f'{args.model1_name} as reference model and {args.model2_name} as target model')

        for trial in range(args.num_trial):
            args.trial = trial
            m1_metrics = pd.read_pickle(f'./{args.assets_dir}/inv_metrics/{args.model1_name}-{args.cap}_{trial}.df')

            save_m2_preds(model2, m1_metrics, tokenizer2, args)
            sh_cap_dict = eval_sh_caps(m1_metrics, sh_cap_dict, args, m1_iid_acc=m1_iid_acc, iid_acc=m2_iid_acc, iid_agg=m2_iid_agg)
            print(f'-'*100)
    
    sh_cap_df = pd.DataFrame.from_dict(sh_cap_dict)

    args.save_dir = f'./paper_figures/sh_plots_{args.model1_name}/{args.cap}'
    print(f'Saving Plots @ {args.save_dir}/{args.save_path}')

    bar_dict = {'model': [], 'metric': [], 'value': []}
    bar_metrics = ['m2_ood_agg', 'hard-scope', 'soft-scope']

    for idx in range(len(sh_cap_df)):
        curr_obj = sh_cap_df.iloc[idx]
        for metric in bar_metrics:
            bar_dict['model'].append(name_dict[curr_obj['m2_models']])
            bar_dict['metric'].append(metric)
            bar_dict['value'].append(curr_obj[f'{metric}'])


    bar_df = pd.DataFrame.from_dict(bar_dict)

    m2_master_list = model2_names
    set_paper_friendly_params()
    
    bar_plot = sns.barplot(data=bar_df, x="metric", y="value", hue="model")
    
    plt.xlabel('')
    plt.ylabel('')

    h, l = bar_plot.get_legend_handles_labels()
    bar_plot.legend(h, l, title="", bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(m2_master_list))

    if args.type == 'depth':
        bar_plot.legend_.remove()

    bar_plot.set_xticks(range(len(np.unique(bar_df.metric))))
    bar_plot.set_xticklabels(['OOD-Agreement', 'Hard-SCoPE', 'Soft-SCoPE'])

    plt.savefig(f'{args.save_dir}/{args.save_path}.pdf', format="pdf", bbox_inches='tight');
    plt.savefig(f'{args.save_dir}/{args.save_path}.png', bbox_inches='tight');


if __name__ == '__main__':
    main()

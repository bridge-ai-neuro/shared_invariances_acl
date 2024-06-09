import tqdm
import numpy as np
import matplotlib.pyplot as plt


CACHE_DIR="PATH/TO/CACHE/DIR"

def model_legends(args):
    
    PATH2NAME = {
        f'bert-base-uncased-{args.dataset}-42_best_model': 'bert-base',
        f'bert-base-uncased-{args.dataset}-43_best_model': 'bert-base',
        f'distilbert-base-uncased-{args.dataset}-42_best_model': 'DistilBERT',
        f'distilbert-base-uncased-{args.dataset}-43_best_model': 'DistilBERT',
        f'gpt2-{args.dataset}-42_best_model': 'GPT-2',
        f'gpt2-{args.dataset}-43_best_model': 'GPT-2',
        f'bert_uncased_L-12_H-768_A-12-{args.dataset}-42_best_model': 'BERT-Base',
        f'bert_uncased_L-8_H-512_A-8-{args.dataset}-42_best_model': 'BERT-Medium',
        f'bert_uncased_L-4_H-512_A-8-{args.dataset}-42_best_model': 'BERT-Small',
        f'bert_uncased_L-4_H-256_A-4-{args.dataset}-42_best_model': 'BERT-Mini',
        f'bert_uncased_L-2_H-128_A-2-{args.dataset}-42_best_model': 'BERT-Tiny'}
    return PATH2NAME


multibert_checkpoints = ['google/multiberts-seed_0-step_0k', 'google/multiberts-seed_0-step_100k', 'google/multiberts-seed_0-step_200k', 'google/multiberts-seed_0-step_300k','google/multiberts-seed_0-step_400k', 'google/multiberts-seed_0-step_500k', 'google/multiberts-seed_0-step_600k', 'google/multiberts-seed_0-step_700k', 'google/multiberts-seed_0-step_800k', 'google/multiberts-seed_0-step_900k', 'google/multiberts-seed_0-step_1000k', 'google/multiberts-seed_0-step_1100k', 'google/multiberts-seed_0-step_1200k', 'google/multiberts-seed_0-step_1300k', 'google/multiberts-seed_0-step_1400k', 'google/multiberts-seed_0-step_1500k', 'google/multiberts-seed_0-step_1600k', 'google/multiberts-seed_0-step_1700k', 'google/multiberts-seed_0-step_1800k', 'google/multiberts-seed_0-step_1900k', 'google/multiberts-seed_0-step_2000k']


def set_paper_friendly_params():
    plt.style.use('seaborn-paper')
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 25
    plt.rcParams['lines.linewidth'] = 4.0
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['lines.markeredgewidth'] = 3
    plt.rcParams['grid.color'] = 'grey'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.25
    plt.rcParams['figure.dpi'] = 75
    plt.rcParams['figure.figsize'] = (12,6)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    print(f'Setting seed as: {seed}')
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def decay(mag, tolerance, cutoff, type_='linear'):

    if type_=='linear':
        fn_eq = lambda x: max(((1-0)/(tolerance-cutoff))*x - ((1-0)/(tolerance-cutoff))*cutoff, 0)
    else:
        raise NotImplementedError

    metric_vals = [None]*len(mag)

    for idx in range(len(mag)):

        if mag[idx]<=tolerance:
            metric_vals[idx]=1
        elif mag[idx]>cutoff:
            metric_vals[idx] = 0
        else:
            metric_vals[idx] = fn_eq(mag[idx])

    return np.array(metric_vals)


def accuracy(model, dataloader, tokenizer, args, return_preds=False):
    tokenize_sent = lambda x: tokenizer(x[args.sentence_key], truncation=True, padding=True, return_tensors='pt')
    
    correct, total = 0,0 
    predictions = []

    for batch in tqdm.tqdm(dataloader):

        sent_token = tokenize_sent(batch).to('cuda')
        label = batch['label'].to('cuda') ## tensor-list of labels in the batch

        output = model(**sent_token) ## tesor-list of outputs in the batch
        pred = output['logits'].argmax(1)

        predictions.extend(pred.tolist())

        correct += (label==pred).sum().item()
        total += len(batch[args.sentence_key])

    acc = (correct/total)

    if return_preds:
        return acc, np.array(predictions)
    
    return acc


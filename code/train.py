## Finetune a model on a given dataset

import argparse
import os

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, AutoConfig)

from utils import CACHE_DIR, seed_everything, multibert_checkpoints

def preprocess_data(dataset, tokenizer, args):
    tokenize_fn = lambda x: tokenizer(x[args.sentence_key], truncation=True)
    encoded_dataset = dataset.map(tokenize_fn, batched=True)

    return encoded_dataset

def compute_metrics(eval_pred):
   load_accuracy = evaluate.load("accuracy")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy}

def finetune_model(encoded_dataset, data_collator, tokenizer, args):

    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels, cache_dir=CACHE_DIR) 
    if args.freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    if tokenizer.pad_token == tokenizer.eos_token:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id

    os.environ["WANDB_PROJECT"] = "PATH/TO/WANDB_DIR"    
    print(f'Saving Model @ {args.run_name}')

    training_args = TrainingArguments(f'finetuned-ckpts/{args.run_name}',
                                        evaluation_strategy='epoch',
                                        save_strategy='epoch',
                                        learning_rate=2e-5,
                                        per_device_train_batch_size=args.batch_size,
                                        per_device_eval_batch_size=args.batch_size,
                                        num_train_epochs=args.epochs,
                                        weight_decay=0.01,
                                        metric_for_best_model='accuracy',
                                        load_best_model_at_end=True,
                                        seed=args.seed,
                                        data_seed=args.seed,
                                        report_to='wandb',
                                        run_name=f'{args.run_name}'
                                        )
    
    print("build trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset[args.train_split],
        eval_dataset=encoded_dataset['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        )

    res = trainer.evaluate()
    print(f'Performance before finetuning: acc: {res["eval_accuracy"]*100:.2f}, loss: {res["eval_accuracy"]:.2f}')
    print(f'-'*100)

    trainer.train()
    trainer.evaluate()
    trainer.save_model(f'finetuned-ckpts/{args.run_name}/best_model')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='sst2')
    parser.add_argument("--sentence_key", default='sentence')
    parser.add_argument("--train_split", default='train')
    parser.add_argument("--eval_split", default='validation')

    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--model_checkpoint", default='distilbert-base-uncased')
    parser.add_argument("--batch_size", type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--freeze_base', action='store_true')
    parser.add_argument('--multibert', action='store_true')
    parser.add_argument("--collection")

    args = parser.parse_args()
    seed_everything(args.seed)

    if args.multibert:
        model_checkpoints = multibert_checkpoints
    else:
        model_checkpoints = [args.model_checkpoint]

    for model_checkpoint in model_checkpoints:
        args.model_checkpoint = model_checkpoint

        if len(args.model_checkpoint.split('/'))>2:
            args.model_checkpoint_name = '_'.join(args.model_checkpoint.split('/'))
        elif len(args.model_checkpoint.split('/'))==2:
            args.model_checkpoint_name = args.model_checkpoint.split('/')[-1]
        else:
            args.model_checkpoint_name = args.model_checkpoint

        if args.freeze_base:
            args.run_name = f'probe_{args.model_checkpoint_name}-{args.dataset}-{args.seed}'        
        else:
            args.run_name = f'{args.model_checkpoint_name}-{args.dataset}-{args.seed}'

        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, cache_dir=CACHE_DIR)
        tokenizer.model_max_length = 512

        if tokenizer.pad_token is None:
            print(f'Adding Padding Token: {tokenizer.eos_token}')
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if args.collection is not None:
            dataset = load_dataset(args.collection, args.dataset, cache_dir=CACHE_DIR)
        else:
            dataset = load_dataset(args.dataset, cache_dir=CACHE_DIR)

        train_val_dataset = dataset['train'].train_test_split(test_size=0.20, stratify_by_column="label")

        train_val_test_dataset = DatasetDict({
        'train': train_val_dataset['train'],
        'valid': train_val_dataset['test'],
        'test': dataset[args.eval_split]})

        encoded_dataset = preprocess_data(train_val_test_dataset, tokenizer, args)
        finetune_model(encoded_dataset, data_collator, tokenizer, args)


if __name__ == '__main__':
    main()

dataset='sst2'
num_labels='2'
sentence_key='sentence'
eval_split='validation'

## Model
model='./finetuned-ckpts/bert-base-uncased-'$dataset'-42/best_model'
model_checkpoint='bert-base-uncased'

cap='taxonomy'
CUDA_VISIBLE_DEVICES=0 python code/generate_perturbations.py --model_checkpoints $model_checkpoint --model_load_paths $model --dataset $dataset  --num_labels $num_labels --sentence_key $sentence_key --eval_split $eval_split --batch_size $batch_size --cap $cap  --assets_dir $assets_dir
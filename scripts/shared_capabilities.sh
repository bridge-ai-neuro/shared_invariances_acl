dataset='sst2'
num_labels='2'
sentence_key='sentence'
eval_split='validation'


assets_dir='assets'
model1='./finetuned-ckpts/bert-base-uncased-'$dataset'-42/best_model'
model1_checkpoint='bert-base-uncased'

## Arch. Differences
echo 'Plotting wrt Arch. Differences'
model2='./finetuned-ckpts/distilbert-base-uncased-'$dataset'-42/best_model,./finetuned-ckpts/gpt2-'$dataset'-42/best_model'
model2_checkpoint='distilbert-base-uncased,gpt2'
save_path='diff_arch'


cap='taxonomy'
CUDA_VISIBLE_DEVICES=1 python code/shared_capabilities.py --dataset $dataset --num_labels $num_labels --batch_size $batch_size --cap $cap --model1_checkpoint $model1_checkpoint  --model1_load_path $model1 --model2_checkpoint $model2_checkpoint --model2_load_path $model2 --save_path $save_path --eval_split $eval_split --sentence_key $sentence_key --assets_dir $assets_dir
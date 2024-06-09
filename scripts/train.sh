dataset='sst2'
num_labels='2'
sentence_key='sentence'
eval_split='validation'

MODEL_CHECKPOINTS=('bert-base-uncased')

for model_ckpt in "${MODEL_CHECKPOINTS[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python code/train.py --seed $seed --dataset $dataset --num_labels $num_labels --sentence_key $sentence_key --eval_split $eval_split --batch_size $batch_size --epochs $epochs --model_checkpoint $model_ckpt
done
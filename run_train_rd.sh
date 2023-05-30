EPOCH=3
MAX_LEN=64
EVALS=10
ACCU=1
NR=101
LR=2e-3
BS=4

DATASET=$1

python main.py \
  --train_file t0-3b-$DATASET-train_set.json \
  --valid_files "dev-sentence:gar-$DATASET-sentence-dev_set.json,dev-answer:gar-$DATASET-sentence-dev_set.json,dev-title:gar-$DATASET-title-dev_set.json" \
  --gpuid 0 \
  --cuda \
  --wandb \
  --wandb_name ear-rd-deberta-v3-base-$DATASET-nr$NR-lr$LR-bs$BS-accu$ACCU \
  --log \
  --epoch $EPOCH \
  --null_rank $NR \
  --batch_size $BS \
  --max_len $MAX_LEN \
  --accumulate_step $ACCU \
  --max_lr $LR \
  --eval_per_epoch $EVALS \
  --fp16 \
  --wtop1 \
  --task_type wtop1 \
  --model_type microsoft/deberta-v3-base

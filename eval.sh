#TOKENIZERS_PARALLELISM=true python eval_e2e.py --train_file gar_title_test_set.json --valid_file gar_title_test_set.json --gpuid 0 --cuda --evaluate --model_pt cache/best-tr-t0gar-models/scorer-title-best.bin --rerank_output rrk-t0gar-title.json --batch_size 1 --bm25_dir output_bm25_gar_title_50 --task_type "" --max_len 256 --fp16 --model_type bert-base-uncased


# python main.py --train_file t0-p1_trivia_train_set.json --valid_files "sentence:gar_sentence_trivia_test_set.json,${TARGET}:gar_${TARGET}_trivia_test_set.json,title:gar_title_trivia_test_set.json,dev-sentence:gar_sentence_trivia_dev_set.json,dev-${TARGET}:gar_${TARGET}_trivia_dev_set.json,dev-title:gar_title_trivia_dev_set.json" --gpuid 0 --cuda --wandb --wandb_name tr-dev-t0gar-trivia-nr$NR-lr$LR-bs$BS-accu$ACCU-ep$EPOCH --loss_type weight-divide --log --epoch $EPOCH --null_rank $NR --batch_size $BS --max_len $MAX_LEN --accumulate_step $ACCU --max_lr $LR --eval_per_epoch $EVALS --fp16 --model_type bert-base-uncased > log.train-dev-t0gar-trivia.nr-$NR.bs-$BS.lr-$LR.accu-$ACCU-ep$EPOCH 2>&1

TARGET=$1
TYPE=$2
DATA=$3
GPU=0
MODEL_DIR=$4
MODEL=$MODEL_DIR/scorer-dev-${TARGET}-best.bin
OUTPUT_RI=output/rrk-gar-${DATA}-${TARGET}-test.rd.json
OUTPUT_RD=output/rrk-gar-${DATA}-${TARGET}-test.ri.json
mkdir -p output

if [ "$TYPE" == "RD" ]; then
    declare -A lengths=( ["nq"]="256" ["trivia"]="192")
    MAX_LEN="${lengths[${DATA}]}"
	VFILE=gar-${DATA}-${TARGET}-test_set.json
	TOKENIZERS_PARALLELISM=true python eval_e2e.py \
    --dataset ${DATA} \
    --valid_file $VFILE \
    --gpuid $GPU \
    --cuda \
    --evaluate \
    --model_pt $MODEL \
    --rerank_output $OUTPUT_RD \
    --batch_size 1 \
    --bm25_dir output_bm25_gar-${TARGET}-${DATA}-test \
    --task_type "wtop1" \
    --wtop1 \
    --max_len $MAX_LEN \
    --fp16 \
    --model_type microsoft/deberta-v3-base
else
	VFILE=gar-${DATA}-${TARGET}-test_set.json
    TOKENIZERS_PARALLELISM=true python eval_e2e.py \
    --dataset ${DATA} \
    --valid_file $VFILE \
    --gpuid $GPU \
    --cuda \
    --evaluate \
    --model_pt $MODEL \
    --rerank_output $OUTPUT_RI \
    --batch_size 1 \
    --bm25_dir output_bm25_gar-${TARGET}-${DATA}-test \
    --task_type "" \
    --max_len 64 \
    --fp16 \
    --model_type microsoft/deberta-v3-base
fi

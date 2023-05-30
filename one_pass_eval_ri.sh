set -e

DATA=$1
MODEL_DIR=$2

FUSE_COMMAND="python round_robin_fuse.py output/fused-gar-${DATA}-test.RI.json"
for TARGET in sentence answer title
do
    echo "Run inference for rrk-gar-${DATA}-${TARGET}-test.RI.json"
    bash eval.sh $TARGET RI $DATA $MODEL_DIR
    OUTPUT="output/rrk-gar-${DATA}-${TARGET}-test.RI.json"
    FUSE_COMMAND="$FUSE_COMMAND $OUTPUT"
    python extract_qas.py output/rrk-gar-${DATA}-${TARGET}-test.RI.json output/query-gar-${DATA}-${TARGET}-test.RI.csv
done
$FUSE_COMMAND
echo "Evaluating fused results: fused-gar-${DATA}-test.RI.json"
if [ "$TRG" != "trec" ]; then
    python eval_dpr.py --retrieval output/fused-gar-${DATA}-test.RI.json --topk 1 5 10 20 50 100 --override
else
    python eval_dpr.py --retrieval output/fused-gar-${DATA}-test.RI.json --topk 1 5 10 20 50 100 --regex --override
fi


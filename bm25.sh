export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/bin/
export JVM_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so

python sparse_retriever.py \
--index_name wikipedia-dpr \
--qa_file "$1" \
--ctx_file ./downloads/data/wikipedia_split/psgs_w100.tsv \
--output_dir $2 \
--n-docs 100 \
--num_threads 32 \
--no_wandb \
--dedup \
--pyserini_cache ./bm25_cache

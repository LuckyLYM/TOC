#!/bin/sh

generate_libsvm_mini_batches_with_format () {
  ./generate_mini_batches.o --input_directory=${3} \
    --output_directory=${2} \
    --output_format=${1} \
    --mini_batch_size=250 \
    --num_mini_batches_per_shard=1000 \
    --start_shard_id=0 \
    --csv_reader=false \
    --end_shard_id=1  \
    --quantization=true \
    --bits=$4 
}

mkdir -p ../data/mnist/mini_batches/
echo "generating toc format"
generate_libsvm_mini_batches_with_format toc ../data/mnist/mini_batches/ ../data/mnist/raw_data $1


#!/bin/sh

### Example usage:
### bash generate_generate_mini_batches_libsvm 100
### Note 100 specifies the number of file shards.

generate_libsvm_mini_batches_with_format () {
  ./generate_mini_batches.o --input_directory=${3} \
    --output_directory=${2} \
    --output_format=${1} \
    --mini_batch_size=250 \
    --num_mini_batches_per_shard=1000 \
    --start_shard_id=0 \
    --csv_reader=false \
    --end_shard_id=$4
}

mkdir -p ../data/mnist/mini_batches/
echo "generating toc format"
generate_libsvm_mini_batches_with_format toc ../data/mnist/mini_batches/ ../data/mnist/shuffled_raw_data $1


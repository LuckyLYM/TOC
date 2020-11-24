#!/bin/sh

generate_label_ordered_data(){
	./similar.o --input_directory=${1} \
		--output_directory=${2} \
		--start_shard_id=0 \
		--end_shard_id=${3}
}


# function: read raw data then sort it accoding to labels
# one sample is stored in a row, adll data are stored in human readable format

mkdir -p ../data/mnist/label_ordered_data/
echo "generate similar data"
generate_similar_data ../data/mnist/shuffled_raw_data ../data/mnist/label_ordered_data $1

#!/bin/sh
# drop cache
echo 3 | tee /proc/sys/vm/drop_caches

epoches=3
shards=1

echo Format:toc
./train_models.o --format=toc --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
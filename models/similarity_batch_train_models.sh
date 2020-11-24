#!/bin/sh

epoches=10
shards=1

echo Format:toc
for i in $(seq 1 10)
do
echo 3 | sudo tee /proc/sys/vm/drop_caches

echo train models with stratified_greedy_batches
./train_models.o --format=toc --model=ann --file_directory=../data/mnist/stratified_greedy_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv

echo train models with local_greedy_batches
./train_models.o --format=toc --model=ann --file_directory=../data/mnist/local_greedy_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv

echo train models with global_greedy_batches
./train_models.o --format=toc --model=ann --file_directory=../data/mnist/global_greedy_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv

echo train models with random mini_batches
./train_models.o --format=toc --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
done
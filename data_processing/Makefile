# -I include dir
# -L lib dir
generate_mini_batches:
	g++ -std=c++14 generate_mini_batches.cc -O3 -o generate_mini_batches.o ../matrix/matrix.cc ../io/reader.cc ../compress/compress_data.h -lm -I /data/yiming/software/include -L /data/yiming/software/lib -lgflags -lglog -lsnappy -lboost_iostreams
label_order:
	g++ -std=c++14 label_order.cc -O3 -o label_order.o -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog
split:
	g++ -std=c++14 split.cc -O3 -o split.o -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog
shuffle_batch:
	g++ -std=c++14 shuffle_batch.cc -O3 -o shuffle_batch.o -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog
shuffle_data:
	g++ -std=c++14 shuffle_data.cc -O3 -o shuffle_data.o -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog
greedy:
	g++ -std=c++14 greedy.cc -O3 -o greedy.o ../matrix/matrix.cc ../io/reader.cc ../compress/compress_data.h -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog -I. -L. -lsnappy -lboost_iostreams
all: generate_mini_batches label_order split shuffle_data shuffle_batch greedy
clean:
	rm -rf *.o *.txt

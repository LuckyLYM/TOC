#include <chrono>
#include <ctime>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <snappy.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../io/reader.h"
#include "../matrix/matrix.h"
#include "../util/gzip.h"

//https://github.com/gflags/gflags/archive/v2.2.2.tar.gz
//https://centos.pkgs.org/7/epel-x86_64/
// functionality: input raw data, output data of various format
using namespace std;

DEFINE_string(input_directory, "", "the input directory path");
DEFINE_string(input_file, "", "the input file path");
DEFINE_string(output_directory, "", "the output directory path");
DEFINE_string(
    output_format, "",
    "The format of the output files. We support mat, csr, csrvi, dvi, toc, "
    "logicalfoc, gzip, and snappy");
DEFINE_int32(mini_batch_size, 0, "# of rows in a mini batch");
DEFINE_int32(label_index, -1, "the index of the label column");
DEFINE_int32(num_mini_batches_per_shard, 0, "# of mini batches in a shard");
DEFINE_int32(start_shard_id, 0, "The starting shard id");
DEFINE_int32(end_shard_id, 120, "The end shard id");
DEFINE_int32(num_read_rows, -1, "the number of rows we read from the file");
// move from reader.cc to this file
DEFINE_double(num_read_columns, 784, "The number of columns in the reader");
DEFINE_bool(
    check_correctness, false,
    "check that the mat equals after serialization and deserialization");
DEFINE_bool(
    csv_reader, true,
    "If true, we use the csv reader to read the "
    "original file name and sample mini batches from it; Otherwise, we read "
    "file shards in libsvm format and generate file shards in different "
    "formats.");
DEFINE_double(scale_factor, 1.0 / 256, "used to scale the data");

DEFINE_bool(
    quantization, false,
    "whether we use data quantization");
DEFINE_int32(bits, 6, "the number bits used for data quantization");

// num_read_columns:
// 784 for MNIST, 3072 for cifar10

// pass parameters according to the specific dataset used
// input_directory, output_directory, num_read_columns, num_mini_batches_per_shard


namespace {

std::vector<std::vector<double>>
scale_dense_mat(std::vector<std::vector<double>> input_dense_mat,
                double scale_factor) {
  if (scale_factor != 0) {
    for (int i = 0; i < input_dense_mat.size(); i++) {
      for (int j = 0; j < input_dense_mat[i].size(); j++) {
        input_dense_mat[i][j] *= scale_factor;
      }
    }
  }
  return std::move(input_dense_mat);
}

std::vector<std::vector<io::sparse_pair>>
scale_sparse_mat(std::vector<std::vector<io::sparse_pair>> input_sparse_mat,
                 double scale_factor) {
  if (scale_factor != 0) {
    for (int i = 0; i < input_sparse_mat.size(); i++) {
      for (int j = 0; j < input_sparse_mat[i].size(); j++) {
        input_sparse_mat[i][j].second *= scale_factor;
      }     
    }
  }
  return std::move(input_sparse_mat);
}

} // namespace

int main(int argc, char **argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(system(("mkdir -p " + FLAGS_output_directory).c_str()) == 0);

  // is csv_reader
  std::unique_ptr<io::Reader> reader;
  if (FLAGS_csv_reader) {
    reader =
        std::make_unique<io::CsvReader>(FLAGS_input_file, FLAGS_label_index);
    reader->read(FLAGS_num_read_rows,FLAGS_num_read_columns);
  }

  // log information
  int nbatch=FLAGS_num_mini_batches_per_shard*FLAGS_end_shard_id;
  int batch_size=FLAGS_mini_batch_size;
  int nbyte=0;
  string form=FLAGS_output_format;


  for (int i = FLAGS_start_shard_id; i < FLAGS_end_shard_id; i++) {
    // read a file shard once

    // we don't use csv_reader in MNIST Experiments
    if (!FLAGS_csv_reader) {
      // Create the libsvm reader.
      // and read from the libsvm file
      string input_file_name =
          FLAGS_input_directory + "/file-" + to_string(i) + ".libsvm";
      reader = std::make_unique<io::LibsvmReader>(input_file_name);
      reader->read(FLAGS_num_read_rows,FLAGS_num_read_columns);
    }

    // output file name
    string file_name;
    if(FLAGS_quantization){
      file_name = FLAGS_output_directory + "/"+to_string(FLAGS_bits)+"-" + to_string(i) + "." +
                         FLAGS_output_format;
    }
    else{
      file_name = FLAGS_output_directory + "/file-" + to_string(i) + "." +
                         FLAGS_output_format;
    }

    fstream shard(file_name, ios::out | ios::binary);
    CHECK(shard);


    // write number of mini_batches in r=this shard
    int num_mini_batches = FLAGS_num_mini_batches_per_shard;
    shard.write((char *)&num_mini_batches, sizeof(int));
    // file.write(s,n)
    // s: pointer to an array of at least n characters
    // n: Number of characters to insert


    for (int j = 0; j < num_mini_batches; j++) {
      const int mini_batch_size = FLAGS_mini_batch_size;
      string format = FLAGS_output_format;

      // Sample the mini batch.
      if (FLAGS_csv_reader) {
        // csv_reader does not support quantization yet
        io::CsvReader *csv_reader = dynamic_cast<io::CsvReader *>(reader.get());
        csv_reader->sample_mini_batch(FLAGS_mini_batch_size);
      } else {
        bool quan=FLAGS_quantization;
        int bits=FLAGS_bits;
        io::LibsvmReader *libsvm_reader =
            dynamic_cast<io::LibsvmReader *>(reader.get());



      // ######################### new feature ############################


            
        libsvm_reader->sample_mini_batch(FLAGS_mini_batch_size, j,quan,bits);
      }



      std::string serialized_string;

      // ######################### new feature ############################
      if (format == "bitoc") {
        core::CompressedMat mat = core::CompressedMat::CreateCompressedMat(
            scale_sparse_mat(*reader->get_sparse_mini_batch(),
                             FLAGS_scale_factor),
            /*init_pairs=*/{},
            /*dim=*/reader->get_num_cols());

            // reach the end of a shard
            // plot some logging information
            if (j==num_mini_batches-1){
              string summary=mat.getSummary();
              double quan_error=reader->get_ave_quan_error();
              string error="bits: "+to_string(FLAGS_bits)+"   quan_error: "+to_string(quan_error)+"\n";
              cout<<summary;
              cout<<error;

              string toc_name = FLAGS_output_directory + "/log.toc";
              fstream f(toc_name, ios::out | ios::app);
              f<<summary;
              f<<error<<endl;
              f.close();
            }



        serialized_string = mat.serialize_as_string();
        // use size() function to get the number of bytes to represent the compressed data

        if (FLAGS_check_correctness) {
          core::CompressedMat other_mat =
              core::CompressedMat::CreateFromString(serialized_string);
          CHECK(mat == other_mat);
        }
      } 

      


      // transform it into a string
      int num_bytes = serialized_string.size();
      nbyte+=num_bytes;
      // write number of bytes
      shard.write((char *)&num_bytes, sizeof(int));
      // write compressed file
      shard.write((char *)&serialized_string[0], num_bytes);
      std::vector<int> mini_batch_labels = *reader->get_mini_batch_labels();


      // we can also compress the labels, but the improvement is not significant
      // write lables in 32-bit int format
      int num_labels = mini_batch_labels.size();
      shard.write((char *)&num_labels, sizeof(int));
      for (int i = 0; i < num_labels; i++) {
        int label = mini_batch_labels[i];
        shard.write((char *)&label, sizeof(int));
      }

    } // end of a mini-batch 
    shard.close();
  } // end of a shard

  double mb=nbyte/1024.0/1024.0;

  string flag="false";
  if(FLAGS_quantization){
    flag="true";
  }

  // write into files
  cout<<"#batch: "<<nbatch<<"   bacth_size: "<<batch_size<<"   format: "<<form<<"   MB: "<<mb<<"   #bits: "<<FLAGS_bits<<"   quantization: "<<flag<<endl;

  // log information of all formats
  string log_name = FLAGS_output_directory + "/log.all";
  fstream f(log_name, ios::out | ios::app);
  f<<"#batch: "<<nbatch<<"   bacth_size: "<<batch_size<<"   format: "<<form<<"   MB: "<<mb<<"   #bits: "<<FLAGS_bits<<"   quantization: "<<flag<<endl;
  f.close();

  return 0;
}

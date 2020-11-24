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

using namespace std;

// the whole file is generated based on generate_mini_batches

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
DEFINE_string(mode,"stratified_greedy","three mode: local greedy, global greedy, and stratified_greedy");

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

  google::InitGoogleLogging(argv[0]);
  CHECK(system(("mkdir -p " + FLAGS_output_directory).c_str()) == 0);

  // is csv_reader
  std::unique_ptr<io::Reader> reader;
  if (FLAGS_csv_reader) {
    reader =
        std::make_unique<io::CsvReader>(FLAGS_input_file, FLAGS_label_index);
    reader->read(FLAGS_num_read_rows);
  }

  for (int i = FLAGS_start_shard_id; i < FLAGS_end_shard_id; i++) {
    // read a file shard once
    if (!FLAGS_csv_reader) {
      // Create the libsvm reader.
      string input_file_name =
          FLAGS_input_directory + "/file-" + to_string(i) + ".libsvm";
      reader = std::make_unique<io::LibsvmReader>(input_file_name);
      reader->read(FLAGS_num_read_rows);
    }

    string file_name = FLAGS_output_directory + "/file-" + to_string(i) + "." +
                       FLAGS_output_format;
    fstream shard(file_name, ios::out | ios::binary);
    CHECK(shard);

    int num_mini_batches = FLAGS_num_mini_batches_per_shard;
    shard.write((char *)&num_mini_batches, sizeof(int));


    // introduce the similarity measure and reorder data here
    // take MNIST for examples we have 250000 samples now

    // added by Yiming
    if(FLAGS_mode=="stratified_greedy")
      reader->stratified_greedy(); 
    else if(FLAGS_mode=="local_greedy")
      reader->local_greedy();
    else if(FLAGS_mode=="global_greedy")
      reader->global_greedy();
    else{
      LOG(INFO)<<"not supported mode "<<FLAGS_mode;
      return 1;
    }

    LOG(INFO)<<FLAGS_mode;



    for (int j = 0; j < num_mini_batches; j++) {
      const int mini_batch_size = FLAGS_mini_batch_size;
      string format = FLAGS_output_format;
      // Sample the mini batch.
      if (FLAGS_csv_reader) {
        io::CsvReader *csv_reader = dynamic_cast<io::CsvReader *>(reader.get());
        csv_reader->sample_mini_batch(FLAGS_mini_batch_size);
      } else {
        io::LibsvmReader *libsvm_reader =
            dynamic_cast<io::LibsvmReader *>(reader.get());
        libsvm_reader->sample_mini_batch(FLAGS_mini_batch_size, j);
      }

      std::string serialized_string;
      if (format == "mat") {  // double type matrix uncompressed form
        core::Mat mat = core::Mat::CreateMat(scale_dense_mat(
            *reader->get_dense_mini_batch(), FLAGS_scale_factor));
        serialized_string = mat.serialize_as_string();
        if (FLAGS_check_correctness) {
          core::Mat other_mat = core::Mat::CreateFromString(serialized_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "toc") {
        core::CompressedMat mat = core::CompressedMat::CreateCompressedMat(
            scale_sparse_mat(*reader->get_sparse_mini_batch(),
                             FLAGS_scale_factor),
            /*init_pairs=*/{},
            /*dim=*/reader->get_num_cols());
        serialized_string = mat.serialize_as_string();
        if (FLAGS_check_correctness) {
          core::CompressedMat other_mat =
              core::CompressedMat::CreateFromString(serialized_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "logicalfoc") {
        core::LogicalCompressedMat mat =
            core::LogicalCompressedMat::CreateLogicalCompressedMat(
                scale_sparse_mat(*reader->get_sparse_mini_batch(),
                                 FLAGS_scale_factor),
                /*init_pairs=*/{},
                /*dim=*/reader->get_num_cols());
        serialized_string = mat.serialize_as_string();
        if (FLAGS_check_correctness) {
          core::LogicalCompressedMat other_mat =
              core::LogicalCompressedMat::CreateFromString(
                  serialized_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "csr") {
        // index stored in uint32 format
        // value stored in double format
        core::CsrMat mat = core::CsrMat::CreateCsrMat(
            scale_sparse_mat(*reader->get_sparse_mini_batch(),
                             FLAGS_scale_factor),
            reader->get_num_cols());
        serialized_string = mat.serialize_as_string();
        if (FLAGS_check_correctness) {
          core::CsrMat other_mat =
              core::CsrMat::CreateFromString(serialized_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "dvi") { // store index and value as string
        core::DviMat mat = core::DviMat::CreateDviMat(scale_dense_mat(
            *reader->get_dense_mini_batch(), FLAGS_scale_factor));
        serialized_string = mat.serialize_as_string();
        if (FLAGS_check_correctness) {
          core::DviMat other_mat =
              core::DviMat::CreateFromString(serialized_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "csrvi") {  
      // store index and value as string. 
      //I think it is reasonable implementation
        core::CsrViMat mat = core::CsrViMat::CreateCsrViMat(
            scale_sparse_mat(*reader->get_sparse_mini_batch(),
                             FLAGS_scale_factor),
            reader->get_num_cols());
        serialized_string = mat.serialize_as_string();
        if (FLAGS_check_correctness) {
          core::CsrViMat other_mat =
              core::CsrViMat::CreateFromString(serialized_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "snappy") { 
        // first transform it into a string, then do snappy compression
        core::Mat mat = core::Mat::CreateMat(scale_dense_mat(
            *reader->get_dense_mini_batch(), FLAGS_scale_factor));
        std::string input_string = mat.serialize_as_string();
        snappy::Compress(input_string.data(), input_string.size(),
                         &serialized_string);
        if (FLAGS_check_correctness) {
          std::string uncompressed_string;
          snappy::Uncompress(serialized_string.data(), serialized_string.size(),
                             &uncompressed_string);
          core::Mat other_mat = core::Mat::CreateFromString(uncompressed_string);
          CHECK(mat == other_mat);
        }
      } else if (format == "gzip") {
        // first transform it into a string, then do gzip compression
        core::Mat mat = core::Mat::CreateMat(scale_dense_mat(
            *reader->get_dense_mini_batch(), FLAGS_scale_factor));
        std::string input_string = mat.serialize_as_string();
        serialized_string = Gzip::compress(input_string);
        if (FLAGS_check_correctness) {
          std::string uncompressed_string = Gzip::decompress(serialized_string);
          core::Mat other_mat = core::Mat::CreateFromString(uncompressed_string);
          CHECK(mat == other_mat);
        }
      }

      // 
      // first transform it into a string
      int num_bytes = serialized_string.size();
      shard.write((char *)&num_bytes, sizeof(int));
      shard.write((char *)&serialized_string[0], num_bytes);
      std::vector<int> mini_batch_labels = *reader->get_mini_batch_labels();
      int num_labels = mini_batch_labels.size();


      // store labels
      // all labels are stored in int format (4 bytes)
      shard.write((char *)&num_labels, sizeof(int));
      for (int i = 0; i < num_labels; i++) {
        int label = mini_batch_labels[i];
        shard.write((char *)&label, sizeof(int));
      }
    }
    shard.close();
  }
  return 0;
}

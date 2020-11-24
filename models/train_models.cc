#include <chrono>
#include <ctime>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <snappy.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../io/reader.h"
#include "../matrix/matrix.h"
#include "../models/mlp_model.h"
#include "../util/gzip.h"

using namespace std;
using namespace io;

DEFINE_string(file_directory, "../../../data/mnist/mini_batches",
              "the directory that contains all the files");
DEFINE_string(test_file, "../../../data/mnist/mnist_test.csv",
              "the file path to the test file");
DEFINE_int32(num_shards, 1, "the number of file shards we read in each epoch");
DEFINE_string(model, "ann", "the ml model");
DEFINE_string(format, "csr", "the format of the mat");
DEFINE_int32(num_cols, 784, "the number of columns in the dataset");
DEFINE_int32(num_outputs, 10, "the number of outputs in the model");
DEFINE_int32(label_index, 0, "the column index of the label");
DEFINE_int32(num_epoches, 10, " the number of epoches");
DEFINE_double(learning_rate, 0.1, " the learning rate");
DEFINE_bool(
    quantization, false,
    "whether we use data quantization");
DEFINE_int32(bits, 6, "the number bits used for data quantization");


// num_cols:
// 784 for MNIST, 3072 for cifar10

// pass parameters according to the specific dataset used
// file_directory, test_file, num_cols




// global variabes for log
double t_read=0;
double t_train=0;
double t_total=0;
double acc=0;



namespace {

  void LoadStringAndLabels(fstream* file_shard, std::string* serialized_str,
      std::vector<int>* labels) {
    int num_bytes = 0;
    int num_labels = 0;
    // read totol number of bytes
    file_shard->read((char*)&num_bytes, sizeof(int));
    serialized_str->resize(num_bytes);
    // read seralized string
    file_shard->read((char*)&(*serialized_str)[0], num_bytes);
    // read number of labels
    file_shard->read((char*)&num_labels, sizeof(int));
    labels->resize(num_labels);
    // read all labels
    file_shard->read((char*)&(*labels)[0], sizeof(int) * num_labels);
  }

}  // namespace



enum CompressionMethod { NONE = 0, GZIP = 1, SNAPPY = 2 };

template<typename T>
// * is pass reference 传引用
void CreateMatAndLabelsFromFileShard(const string& filename, CompressionMethod compression_method,
    std::vector<T>* mats, std::vector<std::vector<int>>* mats_labels) {
    // a vector of matrix, one for each mini_batch


  fstream file_shard(filename, ios::in | ios::binary);
  CHECK(file_shard.is_open());
  int num_mini_batches;
  // read the numebr of mini_batches
  file_shard.read((char *)&num_mini_batches, sizeof(int));
  string serialized_str;
  string uncompressed_str;
  // read each mini_batch
  for (int i = 0; i < num_mini_batches; i++) {
    std::vector<int> labels;
    LoadStringAndLabels(&file_shard, &serialized_str, &labels);
    if (compression_method == NONE) {
      mats->push_back(T::CreateFromString(serialized_str));

    } else if (compression_method == GZIP) {
      mats->push_back(T::CreateFromString(Gzip::decompress(serialized_str)));

    } else if (compression_method == SNAPPY) {

      snappy::Uncompress(serialized_str.data(), serialized_str.size(), &uncompressed_str);
      mats->push_back(T::CreateFromString(uncompressed_str));
    }
    mats_labels->push_back(std::move(labels));
  }


}

template <typename T1, typename T2>
void TrainModel(const std::vector<T1>& mats,
    const std::vector<std::vector<int>>& labels, T2 *model) {
  const int num_mini_batches = mats.size();

  // do model training in mini_batches
  for (int i = 0; i < num_mini_batches; i++) {
    model->TrainModel(mats[i], labels[i]);
  }
}

template <typename T1, typename T2>
void TestModel(const T1 &mat, const T2 &model, const vector<int> &labels) {
  double test_acc=model.ComputeAccuracy(mat, labels);
  //LOG(INFO) << "loss: " << model.ComputeLoss(mat, labels);
  //LOG(INFO) << "accuracy: " << test_acc;
  acc=test_acc;
}



int main(int argc, char **argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  google::SetLogDestination(google::INFO,"../log");
  google::SetStderrLogging(google::INFO);

  // layer_sizes
  model::MlpModel mlp_model({FLAGS_num_cols, 200, 50, FLAGS_num_outputs},
                            FLAGS_learning_rate, /*mini_batch_size=*/250);

  // enumerate epochs
  for (int epoch = 0; epoch < FLAGS_num_epoches; epoch++) {
    auto epoch_start = chrono::system_clock::now();

    // enumerate file_shards

    // change the input file name
    for (int shard_id = 0; shard_id < FLAGS_num_shards; shard_id++) {
      string file_name;

      if(FLAGS_quantization){
        file_name = FLAGS_file_directory + "/"+to_string(FLAGS_bits)+"-" + to_string(shard_id) + "." +FLAGS_format;
      }
      else{
        file_name = FLAGS_file_directory + "/file-" + to_string(shard_id) + "." + FLAGS_format;
      }

      int64_t read_nano_secs = 0;
      int64_t train_nano_secs = 0;

      std::vector<std::vector<int>> mats_labels;

      if (FLAGS_format == "csr") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::CsrMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);
        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();
        TrainModel(mats, mats_labels, &mlp_model);
        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "csrvi") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::CsrViMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();
        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "dvi") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::DviMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "gzip") {
        auto read_start = std::chrono::system_clock::now();

        std::vector<core::Mat> mats;
        CreateMatAndLabelsFromFileShard(file_name, GZIP, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();

      } else if (FLAGS_format == "snappy") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::Mat> mats;
        CreateMatAndLabelsFromFileShard(file_name, SNAPPY, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "mat") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::Mat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();

      // load TOC format training data
      } else if (FLAGS_format == "toc") {
        auto read_start = std::chrono::system_clock::now();

        std::vector<core::CompressedMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();


        auto train_start = std::chrono::system_clock::now();
        TrainModel(mats, mats_labels, &mlp_model);
        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();

      }

      // first-time matrix computation time

      double shard_reading_time=read_nano_secs / (1000.0 * 1000.0 * 1000.0);
      double shard_training_time=train_nano_secs / (1000.0 * 1000.0 * 1000.0);
      double shard_total_time=(read_nano_secs + train_nano_secs) / (1000.0 * 1000.0 * 1000.0);

      t_total+=shard_total_time;
      t_read+=shard_reading_time;
      t_train+=shard_training_time;

      /*
      LOG(INFO) << "epoch_id: " << epoch;
      LOG(INFO) << "shard_id: " << shard_id;
      LOG(INFO) << "reading_time: " << shard_reading_time << " secs";
      LOG(INFO) << "training_time: "<< shard_training_time << " secs";
      LOG(INFO) << "total_time: " << shard_total_time<< " secs";
      */

      // test the acc of model after each epoch
      if (FLAGS_test_file != "") {
        // Read the test data
        CsvReader reader(FLAGS_test_file, FLAGS_label_index);
        reader.read();
        core::Mat mat = core::Mat::CreateMat(*reader.get_dense_mat());
        std::vector<int> labels = *reader.get_labels();
        TestModel(mat, mlp_model, labels);
      }

    } //end of a shard


  }// end of a epoch

  int nepoch=FLAGS_num_epoches;
  int nshard=FLAGS_num_shards;
  string form=FLAGS_format;


  string flag="false";
  if(FLAGS_quantization){
    flag="true";
  }

  cout<<"format: "<<form<<"   #epoch: "<<nepoch<<"   #shard: "<<nshard<<"   #bits: "<<FLAGS_bits<<"   quantization: "<<flag<<endl;
  cout<<"reading_time: "<<t_read<<"   training_time: "<<t_train<<"   total_time: "<<t_total<<"   accuracy: "<<acc<<endl;

  string exp_name = FLAGS_file_directory + "/exp.all";
  fstream f(exp_name, ios::out | ios::app);
  f<<"format: "<<form<<"   #epoch: "<<nepoch<<"   #shard: "<<nshard<<"   #bits: "<<FLAGS_bits<<"   quantization: "<<flag<<endl;
  f<<"reading_time: "<<t_read<<"   training_time: "<<t_train<<"   total_time: "<<t_total<<"   accuracy: "<<acc<<endl;
  f.close();

  return 0;
}

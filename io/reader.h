#ifndef _IO_DATA_READER_H_
#define _IO_DATA_READER_H_

#include <functional>
#include <glog/logging.h>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
using namespace std;



namespace io {

// the sparse format of an attribute
typedef std::pair<int, double> sparse_pair;


template <typename T1, typename T2> struct pair_hash {

  size_t operator()(const std::pair<T1, T2> &my_pair) const {
   
    size_t h1 = std::hash<int>()(my_pair.first);
    size_t h2 = std::hash<double>()(my_pair.second);
   
    // bitwise XOR    why we do this??
    return h1 ^ (h2 << 1);
  }
};


// The abstract reader class.
class Reader {
public:
  // <file_path> specifies the file_path where you read data.
  Reader(std::string file_path) : file_path_(file_path) {}

  // Read the data. <read_rows> specifies the number of rows that we read.
  // If <read_rows> is a negative number, read all the rows. The return value
  // tells if reading is successfully.
  virtual bool read(int read_rows = -1, int read_columns=-1) = 0;

  const std::vector<std::vector<double>> *get_dense_mat() const {
    return &dense_mat_;
  }

  const std::vector<std::vector<double>> *get_dense_mini_batch() const {
    return &dense_mini_batch_;
  }

  const std::vector<std::vector<sparse_pair>> *get_sparse_mat() const {
    return &sparse_mat_;
  }

  const std::vector<std::vector<sparse_pair>> *get_sparse_mini_batch() const {
    return &sparse_mini_batch_;
  }

  const std::vector<int> *get_labels() const { return &labels_; }

  const std::vector<int> *get_mini_batch_labels() const {
    return &mini_batch_labels_;
  }

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

  double get_ave_quan_error()  {return total_quan_error/counter;}

  int get_nbits() {return nbit_;}

 // void shuffle(const vector<int> & new_position, vector<vector<int> > & sf);
  int similarity(int index1,int index2);
  // hamming distance
  int distance(int index1, int index2);
  void update_mat( const vector<vector<int> > & new_position);
  void local_greedy();
  void global_greedy();
  void stratified_greedy();
  void dump(const vector<vector<int>> & new_position);
  bool check_position(const vector<vector<int>> & new_position);
  bool dump_position(const vector<vector<int>> & new_position);
  
  // for data quantization
  double quantization();
  void sign(const vector<vector<sparse_pair>> &sparse, vector<vector<sparse_pair>> &sign);
  double MSE(const vector<vector<sparse_pair>> &sparse);
  void update(vector<vector<sparse_pair>> &sparse_left,vector<vector<sparse_pair>> &sparse_acc, vector<vector<sparse_pair>> &delta);
  double getScale(const vector<vector<sparse_pair>> &sparse,const vector<vector<sparse_pair>> &binary);
  void multiplyScalar(vector<vector<sparse_pair>> &sparse, double value);
  void intialize(const vector<vector<sparse_pair>> &sparse, vector<vector<sparse_pair>> &mat);
  void clamping(vector<vector<sparse_pair>> &sparse_left,vector<vector<sparse_pair>> &sparse_acc);
  int getDistinctValue(const vector<vector<sparse_pair>> &sparse);

protected:
  int num_cols_;
  int num_rows_;
  int nbit_;
  int mini_batch_size_;
  int counter=0; 
  double total_quan_error=0;


  std::string file_path_;
  std::vector<std::vector<double>> dense_mat_;
  std::vector<std::vector<double>> dense_mini_batch_;
  std::vector<std::vector<sparse_pair>> sparse_mat_;
  std::vector<std::vector<sparse_pair>> sparse_mini_batch_;
  std::vector<int> labels_;
  std::vector<int> mini_batch_labels_;

  // ############################### new feature ##################################
  std::vector<std::vector<sparse_pair>> sparse_binary_mini_batch_;
  std::vector<std::vector<double>> dense_binary_mini_batch_;

};

// This class that reads data from csv file into memory
class CsvReader : public Reader {
public:
  // <file_path> specifies the file_path where you read data.
  // <label_index> specifies the col index of the label. If <label_index> is -1,
  // then lables are generated randomly.
  CsvReader(std::string file_path, int label_index = -1)
      : Reader(std::move(file_path)) {
    label_index_ = label_index;
  }

  bool read(int read_rows = -1,int read_columns=-1) override;

  // Randomly sample a mini-batch with <mini_batch_size>.
  void sample_mini_batch(int mini_batch_size);




private:
  int label_index_;
};

// This class reads data from Libsvm format file into memory
class LibsvmReader : public Reader {
public:
  // <file_path> specifies the file_path where you read data.
  LibsvmReader(std::string file_path) : Reader(std::move(file_path)) {}

  // read the data. The return value tells if reading is successfully.
  bool read(int read_rows = -1,int read_columns=-1) override;

  void sample_mini_batch(int mini_batch_size, int mini_batch_id, bool quan, int bits);
};

} // namespace io

#endif // io/data_reader.h

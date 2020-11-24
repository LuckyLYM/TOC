#include "reader.h"

#include <cstdlib>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
//#define DEBUG

using namespace std; // using std

// reader_num_columns is predefined here, we need to change this accordingly
// I need to remove this constraint

#define EPS 1e-6

bool cmp(const std::pair<int,int> &a, const std::pair<int,int> &b){
  return a.first<b.first;
}

namespace io {


// ################################## Implemented by Yiming ##################################
/*
  void shuffle(const vector<int> & new_position, vector<vector<int> > & sf){
    // batch_size 250
    int size=new_position.size();
    int batch_num=sf.size();
    int batch_size=sf[0].size();
    int index=0;
    for(int i=0;i<batch_num;i++){
      for(int j=0;j<batch_size;j++){
        sf[i][j]=new_position[index++];
      }
    }
    random_shuffle(sf.begin(),sf.end());
  }
*/

  // ################################## for batch similarity ##################################

  int Reader::similarity(int index1,int index2){
    vector<double> &d1= dense_mat_[index1];
    vector<double> &d2= dense_mat_[index2];
    int sim=0;
    for(int i=0;i<num_cols_;i++){
      if(d1[i]==d2[i]) sim++;
    }
    return sim;
  }

  // hamming distance
  int Reader::distance(int index1, int index2){
    return num_cols_-similarity(index1,index2);
  }

/*
  void Reader::update_Mat( const vector<int> & new_position){
    vector<vector<sparse_pair> > new_sparse_mat(new_position.size());
    vector<vector<double> > new_dense_mat(new_position.size());
    for(int i=0;i<new_position.size();i++){
      // same question no "=" operator
      //new_sparse_mat[i].assign(sparse_mat_[new_position[i]].begin(),sparse_mat_[new_position[i]].end());
      //new_dense_mat[i].assign(dense_mat_[new_position[i]].begin(),dense_mat_[new_position[i]].end());
      new_sparse_mat[i]=sparse_mat_[new_position[i]];
      new_dense_mat[i]=dense_mat_[new_position[i]];
    }
    LOG(INFO)<<"sparse_mat_swap";
    sparse_mat_.swap(new_sparse_mat);
    LOG(INFO)<<"dense_mat_swap";
    dense_mat_.swap(new_dense_mat);
    LOG(INFO)<<"swap succeed";
  }
*/
  void Reader::update_mat( const vector<vector<int> > & new_position){

    vector<vector<sparse_pair> > new_sparse_mat(num_rows_);
    vector<vector<double> > new_dense_mat(num_rows_);
    vector<int> new_labels(num_rows_);

    int num=0;
    for(int i=0;i<new_position.size();i++){
      for(int j=0;j<new_position[i].size();j++){
        new_sparse_mat[num]=sparse_mat_[new_position[i][j]];
        new_dense_mat[num]=dense_mat_[new_position[i][j]];
        new_labels[num++]=labels_[new_position[i][j]];
      }
    }
    sparse_mat_.swap(new_sparse_mat);
    dense_mat_.swap(new_dense_mat);
    labels_.swap(new_labels);
  }

  // only support libsvm format now
  // write reordered samples into file
  void Reader::dump(const vector<vector<int>> & new_position){ 
    ifstream file(file_path_.c_str());
    CHECK(file.is_open());
    string line;
    int num_rows = 0;
    vector<string> order(num_rows_);
    vector<string> new_order(num_rows_);
    int row=0;
    while (getline(file, line)) {
      order[row++]=line;
    }

    int num=0;
    for(int i=0;i<new_position.size();i++){
      for(int j=0;j<new_position[i].size();j++){
        new_order[num++]=order[new_position[i][j]];
      }
    }

    string output_file_path="file-0.libsvm";   // maybe more flexible here
    fstream output_file(output_file_path.c_str(),ios::out);
    for(int i=0;i<num_rows_;i++){
      output_file<<new_order[i]<<"\n";
    }
    output_file.close();
  }

  bool Reader::check_position(const vector<vector<int>> & new_position){
    vector<int> checklist(num_rows_,0);
    for(int i=0;i<new_position.size();i++){
      for(int j=0;j<new_position[i].size();j++){
        int p=new_position[i][j];
        checklist[p]++;
      }
    }
    for(int i=0;i<num_rows_;i++){
      if(checklist[i]!=1) return false;
    }
    return true;
  }

  bool Reader::dump_position(const vector<vector<int>> & new_position){
    string output_file_path="position";   // maybe more flexible here    
    fstream output_file(output_file_path.c_str(),ios::out);

    for(int i=0;i<new_position.size();i++){
      for(int j=0;j<new_position[i].size();j++){
        output_file<< new_position[i][j]<<endl;
      }
    }    
    output_file.close();
    return true;
  }

  void Reader::global_greedy(){
    // It is a small experiments, still try to pack samples of the same labels into one mini-batch.

    int batch_size=250;
    int batch_num=1000;
    
    vector<bool> packed(num_rows_,false);

    // batch_num and batch_size
    vector<vector<int> > new_position(batch_num,vector<int>(batch_size,0));

    int batch_index=0;
    int sample_index=0;
    int num=0;

    for(int i=0;i<num_rows_;i++){

      if(packed[i]==true) continue;
      new_position[batch_index][sample_index++]=i;
      packed[i]=true;
      num++;

      // initiate candidate list
      int candidate_num=0;
      vector<pair<int,int> > candidate_list(num_rows_-num);

      // search neighbors
      for(int j=0;j<num_rows_;j++){
        if(packed[j]) continue;
        int dis=distance(i,j);
        candidate_list[candidate_num++]=make_pair(dis,j); // (distance,index)
      }

      // sort neighbors according to distance
      sort(candidate_list.begin(),candidate_list.end(),cmp);

      // update new_position list  // found a bug here

      for(int j=0;j<batch_size-1;j++){
        new_position[batch_index][sample_index++]=candidate_list[j].second;
        packed[candidate_list[j].second]=true;
      }
      batch_index++;
      sample_index=0;
      num+=batch_size-1;
      if(batch_index%100==0){
        cout<<"batch_index "<<batch_index<<endl;
      }

      if(num==num_rows_) break;
    } 


    random_shuffle(new_position.begin(),new_position.end());
    CHECK(check_position(new_position));
    update_mat(new_position);
  }


  void Reader:: stratified_greedy(){
    // It is a small experiments, try to pack samples of different labels into one mini-batch.

    int batch_size=250;
    int batch_num=1000;

    int cluster_size=25000;
    int cluster_num=10;  // for MNIST
    int begin,end;

    
    vector<bool> packed(num_rows_,false);
    vector<vector<int> > new_position(batch_num,vector<int>(batch_size,0));

    int batch_index=0;
    int sample_index=0;
    
    // using a greedy stratified samping strategy
    for(batch_index=0;batch_index<batch_num;batch_index++){

      sample_index=0;

      // sample from a cluster
      for(int i=0;i<cluster_num;i++){
        int begin=i*cluster_size;
        int end=(i+1)*cluster_size;
        int j,head;

        int candidate_num=0;                                // hard coded number
        vector<pair<int,int> > candidate_list(cluster_size-batch_index*25-1);

        // find head point
        for(j=begin;j<end;j++){
          if(packed[j]) continue;
          new_position[batch_index][sample_index++]=j;
          packed[j]=true;
          break;
        }

        head=j;
        
        // find candiate neighbors
        for(j=j+1;j<end;j++){          
          if(packed[j]) continue;
          int dis=distance(head,j);
          candidate_list[candidate_num++]=make_pair(dis,j); // (distance,index)
        }

        // sort neighbors according to distance
        sort(candidate_list.begin(),candidate_list.end(),cmp);

        // hard coded number
        for(int j=0;j<25-1;j++){
          new_position[batch_index][sample_index++]=candidate_list[j].second;
          packed[candidate_list[j].second]=true;
        }
      }

      if(batch_index%100==0){
        cout<<"batch index "<<batch_index<<endl;
      }
    }

    random_shuffle(new_position.begin(),new_position.end());
    CHECK(check_position(new_position));
    update_mat(new_position);
  }

  void Reader::local_greedy(){
    // It is a small experiments, still try to pack samples of the same labels into one mini-batch.
    int batch_size=250;
    int batch_num=1000;

    int cluster_size=25000;
    int cluster_num=10;  // for MNIST
    int begin,end;

    
    vector<bool> packed(num_rows_,false);
    //vector<int> new_position(num_rows_); // index-value: new position, original position  old version definition
    // batch_num and batch_size
    vector<vector<int> > new_position(batch_num,vector<int>(batch_size,0));

    int batch_index=0;
    int sample_index=0;
    int num=0;

    for (int index = 0; index <cluster_num ; ++index)
    { 
      begin=index*cluster_size;
      end=(index+1)*cluster_size;
    
      cout<<"begin cluster "<<index<<endl;

      // get a cluster
      for(int i=begin;i<end;i++){
        if(packed[i]==true) continue;
        new_position[batch_index][sample_index++]=i;
        packed[i]=true;
        num++;

        // initiate candidate list
        int candidate_num=0;
        vector<pair<int,int> > candidate_list(end-num);

        // search neighbors
        for(int j=begin;j<end;j++){
          if(packed[j]) continue;
          int dis=distance(i,j);
          candidate_list[candidate_num++]=make_pair(dis,j); // (distance,index)
        }

        // sort neighbors according to distance
        sort(candidate_list.begin(),candidate_list.end(),cmp);

        // update new_position list  // found a bug here
        for(int j=0;j<batch_size-1;j++){
          new_position[batch_index][sample_index++]=candidate_list[j].second;
          packed[candidate_list[j].second]=true;
        }
        batch_index++;
        sample_index=0;
        num+=batch_size-1;
        if(num==end) break;
      } // end a cluster

    }

    random_shuffle(new_position.begin(),new_position.end());
    CHECK(check_position(new_position));
    update_mat(new_position);
  }
























  // ################################## for data quantization ##################################

  void Reader::sign( const vector<vector<sparse_pair>> &sparse, vector<vector<sparse_pair>> &sign){
    int ncol=sparse.size();
    for(int i=0;i<ncol;i++){
      int n2=sparse[i].size();
      vector<sparse_pair> new_row(n2);

      for(int j=0;j<n2;j++){
        int index=sparse[i][j].first;
        double value=sparse[i][j].second;
        if(value<0){
          new_row[j]=make_pair(index,-1);
        }
        else{
          new_row[j]=make_pair(index,1);
        }
      }

      sign.push_back(new_row);
    }
  }

  void Reader::multiplyScalar(vector<vector<sparse_pair>> &sparse, double value){
    int ncol=sparse.size();
    for(int i=0;i<ncol;i++){
      for(int j=0;j<sparse[i].size();j++){
        sparse[i][j].second=sparse[i][j].second*value;
      }
    }
  }


  // the two mats are of the same shape
  double Reader::getScale(const vector<vector<sparse_pair>> &sparse,const vector<vector<sparse_pair>> &binary){
    int ncol=sparse.size();
    double sum=0;
    for(int i=0;i<ncol;i++){
      for(int j=0;j<sparse[i].size();j++){
        sum+=sparse[i][j].second*binary[i][j].second;
      }
    }
    int n=mini_batch_size_*num_cols_;
    return sum/n;
  }


  void Reader::update(vector<vector<sparse_pair>> &sparse_left,vector<vector<sparse_pair>> &sparse_acc, vector<vector<sparse_pair>> &delta){
    int ncol=delta.size();
    for(int i=0;i<ncol;i++){
      for(int j=0;j<delta[i].size();j++){
        sparse_left[i][j].second-=delta[i][j].second;
        sparse_acc[i][j].second+=delta[i][j].second;
      }
    }
  }

  // clamping for sparse_acc, and correct the error term
  void Reader::clamping(vector<vector<sparse_pair>> &sparse_left,vector<vector<sparse_pair>> &sparse_acc){
    int ncol=sparse_left.size();
    for(int i=0;i<ncol;i++){
      for(int j=0;j<sparse_left[i].size();j++){
        double sum=sparse_left[i][j].second+sparse_acc[i][j].second;
        double clamped=max(0.0,min(255.0,sparse_acc[i][j].second));
        sparse_left[i][j].second= sum-clamped;
        sparse_acc[i][j].second=clamped;
      }
    }
  }

  double Reader::MSE(const vector<vector<io::sparse_pair>> &sparse){
    int ncol=sparse.size();
    double sum=0;
    for(int i=0;i<ncol;i++){
      for(int j=0;j<sparse[i].size();j++){
        double value=sparse[i][j].second;
        sum+=value*value;
      }
    }
    int n=mini_batch_size_*num_cols_;
    return sum/256.0/256.0/n;
    // assume 256 is the scaling factor
  }

  // use #bits to do data quantization for the sparse_mat
  // return the mean squared error

  void Reader::intialize(const vector<vector<sparse_pair>> &sparse, vector<vector<sparse_pair>> &mat){
    int ncol=sparse.size();
    for(int i=0;i<ncol;i++){
      int n2=sparse[i].size();
      vector<sparse_pair> new_row(n2);
      for(int j=0;j<n2;j++){
        new_row[j].first=sparse[i][j].first;
        new_row[j].second=0;
      }

      mat.push_back(new_row);
    }
  }

  int Reader::getDistinctValue(const vector<vector<sparse_pair>> &sparse){
    unordered_map<double, int> distinct_values;
    int index=0;
    int ncol=sparse.size();
    for(int i=0;i<ncol;i++){
      int n2=sparse[i].size();
      for(int j=0;j<n2;j++){
        double value=sparse[i][j].second;
        if (distinct_values.find(value) == distinct_values.end()) {
          distinct_values[value] = index++;
        }      
         
      }
    }
    return index;
  }




  double Reader::quantization(){
    // nbit_,num_rows_,num_cols_ ;

    // for debugging
    /*
    cout<<"sparse_mini_batch_: "<<sparse_mini_batch_.size()<<endl;
    for(int i=0;i<sparse_mini_batch_.size();i++){
      for(int j=0;j<sparse_mini_batch_[i].size();j++){

        int index=sparse_mini_batch_[i][j].first;
        double value=sparse_mini_batch_[i][j].second;

        cout<<"i: "<<i<<" j: "<<j<<" index: "<<index<<" value: "<<value<<endl;

      }
    }
    */



    vector<vector<sparse_pair> > & sparse_left=sparse_mini_batch_;
    vector<vector<sparse_pair> > sparse_acc;    //todo a function to intialize this vector
    intialize(sparse_left,sparse_acc);

    // log info
    #ifdef DEBUG
    if(counter==1000){
      cout<<"#bits: "<<nbit_<<endl;
      int num_sparse=getDistinctValue(sparse_left);
      cout<<"#distinct_values (sparse_mini_batch): "<<num_sparse<<endl;
    }
    #endif

    for(int i=0;i<nbit_;i++){
      vector<std::vector<sparse_pair>> binary_mat;  // intialize
      sign(sparse_left,binary_mat);
      double scale=getScale(sparse_left,binary_mat);

      multiplyScalar(binary_mat,scale);
      vector<std::vector<sparse_pair>> & delta=binary_mat;
      update(sparse_left,sparse_acc,delta);

      // log info
      #ifdef DEBUG
      if (counter==1000){
        cout<<"iteration: "<<i<<"   scaling factor: "<<scale<<endl;
        int num_binary=getDistinctValue(binary_mat);
        cout<<"iteration: "<<i<< "   #distinct_values (binary_mat): "<<num_binary<<endl;
        int num_acc=getDistinctValue(sparse_acc);
        cout<<"iteration: "<<i<< "   #distinct_values (sparse_acc): "<<num_acc<<endl;
      }
      #endif
    }

    // value clamping after data quantization
    clamping(sparse_left,sparse_acc);
    
    // log info
    #ifdef DEBUG
    if (counter==1000){
      int num_clamping=getDistinctValue(sparse_acc);
      cout<<"after clamping: #distinct_values (sparse_acc): "<<num_clamping<<endl;
    }
    #endif

    double error=MSE(sparse_left);
    total_quan_error+=error;
    // update quantization error


    //cout<<"quantization error: "<<error<<endl;

    //cout<<"before loop"<<endl;
    // reconstructs the dense_mat and the sparse_mat from the sparse_acc
    // why I see somevalue like 388??
    // how to correct this?? maybe we need sth like min(255,x)
    // make sure the logic is correct tonight
    for (int i = 0; i < mini_batch_size_; i++) {
      for (int j = 0; j < sparse_mini_batch_[i].size(); j++) {

        //cout<<"i: "<<i<<" j: "<<j<<endl;

        int index=sparse_acc[i][j].first;
        double value=sparse_acc[i][j].second;

        //cout<<"index: "<<index<<" value: "<<value<<endl;

        sparse_mini_batch_[i][j].second=value;
        dense_mini_batch_[i][index] = value;
      }
    }
    //cout<<"after loop"<<endl;

    return error;
  }

// ################################## Implemented by Yiming ##################################










  // following are the original files

  bool CsvReader::read(int read_rows, int read_columns) {
    ifstream file(file_path_.c_str());
    CHECK(file.is_open());
    string line;
    int num_rows = 0;
    // constructs the dense mat
    while (getline(file, line)) {
      if (num_rows++ == read_rows)
        break;

      vector<double> vec;
      vector<sparse_pair> sparse_vec; // pair<int, double>
      stringstream ss(line);

      double val;
      int idx = 0;

      while (ss >> val) {
        if (idx == label_index_) {
          labels_.push_back((int)val);
        } 
        else {
          vec.push_back(val);
        }
        idx++;
        // ignore comma and space while reading
        while (ss.peek() == ' ' || ss.peek() == ',') {
          ss.ignore();
        }
      }
      dense_mat_.push_back(vec);
    }

    num_rows_ = dense_mat_.size();
    num_cols_ = dense_mat_[0].size();

    // constructs the sparse mat
    for (int i = 0; i < num_rows_; i++) {
      vector<sparse_pair> sparse_vec;
      for (int j = 0; j < num_cols_; j++) {
        if (fabs(dense_mat_[i][j]) > EPS) {
          sparse_vec.push_back(make_pair(j, dense_mat_[i][j]));
        }
      }
      sparse_mat_.push_back(std::move(sparse_vec));
    }
    // randomly generates the labels if <label_index_> is -1.
    if (label_index_ == -1) {
      for (int i = 0; i < num_rows_; i++) {
        labels_.push_back(rand() % 2);
      }
    }

    file.close();
    return true;
  }

  // random sample
  void CsvReader::sample_mini_batch(int mini_batch_size) {
    dense_mini_batch_.clear();    // vector<vector<double>>
    sparse_mini_batch_.clear();   // vector<vector<pair>>
    mini_batch_labels_.clear();


    for (int i = 0; i < mini_batch_size; i++) {
      int row_index = rand() % num_rows_;
      dense_mini_batch_.push_back(dense_mat_[row_index]);
      sparse_mini_batch_.push_back(sparse_mat_[row_index]);
      mini_batch_labels_.push_back(labels_[row_index]);
    }
    return;
  }


// get a sparse matrix and a dense matrix
  bool LibsvmReader::read(int read_rows, int read_columns) {
    ifstream file(file_path_.c_str());
    CHECK(file.is_open());
    string line;
    int num_rows = 0;

    // libsvm format
    // each row: label <index,value> pairs


    // read and construct sparse mat
    while (getline(file, line)) {
      if (num_rows++ == read_rows)
        break;
      stringstream ss(line);
      string val;
      ss >> val;
      // read label
      labels_.push_back(stoi(val));
      // read <index,value> pair   // sparse_pair<int, double>
      vector<sparse_pair> sparse_vec;  

      while (ss >> val) {
        std::size_t delim = val.find(':');
        int pos = stoi(val.substr(0, delim));
        if (pos >= read_columns)
          continue;
        double value = stod(val.substr(delim + 1, val.length()));
        sparse_vec.push_back(make_pair(pos, value));
      }
      sparse_mat_.push_back(std::move(sparse_vec));
    }


    num_rows_ = sparse_mat_.size();
    num_cols_ = read_columns;

    // constructs the dense mat
    for (int i = 0; i < num_rows_; i++) {
      // Initialize an all zero vector with dimension 
      std::vector<double> dense_row(read_columns, 0);
      for (int j = 0; j < sparse_mat_[i].size(); j++) {
        dense_row[sparse_mat_[i][j].first] = sparse_mat_[i][j].second;
      }
      dense_mat_.push_back(dense_row);
    }

    file.close();
    return true;
  }

// sequentially read a mini batch
// not random sample
  void LibsvmReader::sample_mini_batch(int mini_batch_size, int mini_batch_id, bool quan, int bits) {
    mini_batch_size_=mini_batch_size;
    nbit_=bits;
    dense_mini_batch_.clear();
    sparse_mini_batch_.clear();
    mini_batch_labels_.clear();
    counter++;
    const int start_row_index = mini_batch_id * mini_batch_size;
    const int end_row_index = (mini_batch_id + 1) * mini_batch_size;
    for (int i = start_row_index; i < end_row_index; i++) {
      dense_mini_batch_.push_back(dense_mat_[i]);
      sparse_mini_batch_.push_back(sparse_mat_[i]);
      mini_batch_labels_.push_back(labels_[i]);
    }

    if (quan){
      quantization();
    }

  }

} // namespace io

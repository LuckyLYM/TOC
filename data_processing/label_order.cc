#include <chrono>
#include <ctime>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>


using namespace std;


// this version only supports libsvm format and MNIST datasets
DEFINE_string(input_directory, "", "the input directory path");
DEFINE_string(input_file, "", "the input file path");
DEFINE_string(output_directory, "", "the output directory path");
DEFINE_int32(label_index, -1, "the index of the label column");
DEFINE_int32(start_shard_id, 0, "The starting shard id");
DEFINE_int32(end_shard_id, 120, "The end shard id");
DEFINE_int32(num_read_rows, -1, "the number of rows we read from the file");

int main(int argc, char **argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
    CHECK(system(("mkdir -p " + FLAGS_output_directory).c_str()) == 0);
    int num_lables=10;

    for (int i = FLAGS_start_shard_id; i < FLAGS_end_shard_id; i++) {
        vector<int> label_counter(num_lables,0);

        LOG(INFO) << "shard_id: " << i;
        vector<vector<string> > list (num_lables);
        string input_file_name = FLAGS_input_directory + "/file-" + to_string(i) + ".libsvm";
        ifstream file(input_file_name.c_str());
        string line;
        int num_rows=0;
          while (getline(file, line)) {
            if (num_rows++ == FLAGS_num_read_rows)
              break;
			if (num_rows%10000==0){
				LOG(INFO)<<"Read row numbers: "<<num_rows;
			}
            stringstream ss(line);
            string val;
            ss >> val;
			int label=stoi(val);
            label_counter[label]++;
            list[label].push_back(line);          
          }


        string output_file_name = FLAGS_output_directory + "/file-" + to_string(i)+".libsvm";
        fstream shard(output_file_name, ios::out);
        CHECK(shard);
		LOG(INFO)<<"start writing file";
        for(int j=0;j<num_lables;j++){
            int size=label_counter[j];
            LOG(INFO) << "label: " << j<<" num: "<< size;
            LOG(INFO) << "writing label " << j;
            for (int z = 0; z < size; z++)
            {
                shard<<list[j][z]<<"\n";
            }
        }
        shard.close();
    }
    return 0;
}

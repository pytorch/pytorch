/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <array>
#include <fstream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>

#include "caffe2/core/common.h"
#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/core/logging.h"

C10_DEFINE_string(input_folder, "", "The input folder name.");
C10_DEFINE_string(output_train_db_name, "", "The output training db name.");
C10_DEFINE_string(output_test_db_name, "", "The output testing db name.");
C10_DEFINE_string(db, "leveldb", "The db type.");
C10_DEFINE_bool(
    is_cifar100,
    false,
    "If set, convert cifar100. Otherwise do cifar10.");

namespace caffe2 {

using std::stringstream;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = kCIFARSize * kCIFARSize * 3;
const int kCIFAR10BatchSize = 10000;
const int kCIFAR10TestDataSize = 10000;
const int kCIFAR10TrainBatches = 5;

const int kCIFAR100TrainDataSize = 50000;
const int kCIFAR100TestDataSize = 10000;

void ReadImage(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  if (FLAGS_is_cifar100) {
    // Skip the coarse label.
    file->read(&label_char, 1);
  }
  file->read(&label_char, 1);
  *label = label_char;
  // Yes, there are better ways to do it, like in-place swap... but I am too
  // lazy so let's just write it in a memory-wasteful way.
  std::array<char, kCIFARImageNBytes> channel_first_storage;
  file->read(channel_first_storage.data(), kCIFARImageNBytes);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < kCIFARSize * kCIFARSize; ++i) {
      buffer[i * 3 + c] =
          channel_first_storage[c * kCIFARSize * kCIFARSize + i];
    }
  }
  return;
}

void WriteToDB(const string& filename, const int num_items,
                    const int& offset, db::DB* db) {
  TensorProtos protos;
  TensorProto* data = protos.add_protos();
  TensorProto* label = protos.add_protos();
  data->set_data_type(TensorProto::BYTE);
  data->add_dims(kCIFARSize);
  data->add_dims(kCIFARSize);
  data->add_dims(3);
  label->set_data_type(TensorProto::INT32);
  label->add_dims(1);
  label->add_int32_data(0);

  LOG(INFO) << "Converting file " << filename;
  std::ifstream data_file(filename.c_str(),
      std::ios::in | std::ios::binary);
  CAFFE_ENFORCE(data_file, "Unable to open file ", filename);
  char str_buffer[kCIFARImageNBytes];
  int label_value;
  string serialized_protos;
  std::unique_ptr<db::Transaction> transaction(db->NewTransaction());
  for (int itemid = 0; itemid < num_items; ++itemid) {
    ReadImage(&data_file, &label_value, str_buffer);
    data->set_byte_data(str_buffer, kCIFARImageNBytes);
    label->set_int32_data(0, label_value);
    protos.SerializeToString(&serialized_protos);
    snprintf(str_buffer, kCIFARImageNBytes, "%05d",
        offset + itemid);
    transaction->Put(string(str_buffer), serialized_protos);
  }
}

void ConvertCIFAR() {
  std::unique_ptr<db::DB> train_db(
      db::CreateDB(FLAGS_db, FLAGS_output_train_db_name, db::NEW));
  std::unique_ptr<db::DB> test_db(
      db::CreateDB(FLAGS_db, FLAGS_output_test_db_name, db::NEW));

  if (!FLAGS_is_cifar100) {
    // This is cifar 10.
    for (int fileid = 0; fileid < kCIFAR10TrainBatches; ++fileid) {
      stringstream train_file;
      train_file << FLAGS_input_folder << "/data_batch_" << fileid + 1
                 << ".bin";
      WriteToDB(train_file.str(), kCIFAR10BatchSize,
                fileid * kCIFAR10BatchSize, train_db.get());
    }
    stringstream test_file;
    test_file << FLAGS_input_folder << "/test_batch.bin";
    WriteToDB(test_file.str(), kCIFAR10TestDataSize, 0, test_db.get());
  } else {
    // This is cifar 100.
    stringstream train_file;
    train_file << FLAGS_input_folder << "/train.bin";
    WriteToDB(train_file.str(), kCIFAR100TrainDataSize, 0, train_db.get());
    stringstream test_file;
    test_file << FLAGS_input_folder << "/test.bin";
    WriteToDB(test_file.str(), kCIFAR100TestDataSize, 0, test_db.get());
  }
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ConvertCIFAR();
  return 0;
}

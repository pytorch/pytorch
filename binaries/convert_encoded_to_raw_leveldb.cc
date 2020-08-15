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

// This script converts an image dataset to leveldb.
//
// FLAGS_input_folder is the root folder that holds all the images, and
// FLAGS_list_file should be a list of files as well as their labels, in
// the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <opencv2/opencv.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <memory>
#include <random>
#include <string>

#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/core/logging.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

C10_DEFINE_string(input_db_name, "", "The input image file name.");
C10_DEFINE_string(output_db_name, "", "The output training leveldb name.");
C10_DEFINE_bool(color, true, "If set, load images in color.");
C10_DEFINE_int(
    scale,
    256,
    "If FLAGS_raw is set, scale all the images' shorter edge to the given "
    "value.");
C10_DEFINE_bool(warp, false, "If warp is set, warp the images to square.");

namespace caffe2 {


void ConvertToRawDataset(
    const string& input_db_name, const string& output_db_name) {
  // input leveldb
  std::unique_ptr<leveldb::DB> input_db;
  LOG(INFO) << "Opening input leveldb " << input_db_name;
  {
    leveldb::Options options;
    options.create_if_missing = false;
    leveldb::DB* db_temp;
    leveldb::Status status = leveldb::DB::Open(
        options, input_db_name, &db_temp);
    CAFFE_ENFORCE(status.ok(), "Failed to open leveldb ", input_db_name, ".");
    input_db.reset(db_temp);
  }

  // output leveldb
  std::unique_ptr<leveldb::DB> output_db;
  std::unique_ptr<leveldb::WriteBatch> batch;
  LOG(INFO) << "Opening leveldb " << output_db_name;
  {
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    leveldb::DB* db_temp;
    leveldb::Status status = leveldb::DB::Open(
        options, output_db_name, &db_temp);
    CAFFE_ENFORCE(
        status.ok(),
        "Failed to open leveldb ",
        output_db_name,
        ". Is it already existing?");
    output_db.reset(db_temp);
  }
  batch.reset(new leveldb::WriteBatch());

  TensorProtos input_protos;
  TensorProtos output_protos;
  TensorProto* data = output_protos.add_protos();
  TensorProto* label = output_protos.add_protos();
  data->set_data_type(TensorProto::BYTE);
  data->add_dims(0);
  data->add_dims(0);
  if (FLAGS_color) {
    data->add_dims(3);
  }
  string value;

  unique_ptr<leveldb::Iterator> iter;
  iter.reset(input_db->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  int count = 0;
  for (; iter->Valid(); iter->Next()) {
    CAFFE_ENFORCE(input_protos.ParseFromString(iter->value().ToString()));
    label->CopyFrom(input_protos.protos(1));
    const string& encoded_image = input_protos.protos(0).string_data(0);
    int encoded_size = encoded_image.size();
    cv::Mat img = cv::imdecode(
        cv::Mat(
            1, &encoded_size, CV_8UC1, const_cast<char*>(encoded_image.data())),
        FLAGS_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    cv::Mat resized_img;
    int scaled_width, scaled_height;
    if (FLAGS_warp) {
      scaled_width = FLAGS_scale;
      scaled_height = FLAGS_scale;
    } else if (img.rows > img.cols) {
      scaled_width = FLAGS_scale;
      scaled_height = static_cast<float>(img.rows) * FLAGS_scale / img.cols;
    } else {
      scaled_height = FLAGS_scale;
      scaled_width = static_cast<float>(img.cols) * FLAGS_scale / img.rows;
    }
    cv::resize(img, resized_img, cv::Size(scaled_width, scaled_height), 0, 0,
                 cv::INTER_LINEAR);
    data->set_dims(0, scaled_height);
    data->set_dims(1, scaled_width);
    DCHECK(resized_img.isContinuous());
    data->set_byte_data(
        resized_img.ptr(),
        scaled_height * scaled_width * (FLAGS_color ? 3 : 1));
    output_protos.SerializeToString(&value);
    // Put in db
    batch->Put(iter->key(), value);
    if (++count % 1000 == 0) {
      output_db->Write(leveldb::WriteOptions(), batch.get());
      batch.reset(new leveldb::WriteBatch());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    output_db->Write(leveldb::WriteOptions(), batch.get());
  }
  LOG(INFO) << "Processed a total of " << count << " files.";
}

}  // namespace caffe2


int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ConvertToRawDataset(FLAGS_input_db_name, FLAGS_output_db_name);
  return 0;
}

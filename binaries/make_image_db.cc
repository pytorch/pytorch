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

// This script converts an image dataset to a database.
//
// caffe2::FLAGS_input_folder is the root folder that holds all the images
//
// caffe2::FLAGS_list_file is the path to a file containing a list of files
// and their labels, as follows:
//
//   subfolder1/file1.JPEG 7
//   subfolder1/file2.JPEG 7
//   subfolder2/file1.JPEG 8
//   ...
//

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <thread>

#include "caffe2/core/common.h"
#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
CAFFE2_DEFINE_string(input_folder, "", "The input image file name.");
CAFFE2_DEFINE_string(
    list_file,
    "",
    "The text file containing the list of images.");
CAFFE2_DEFINE_string(output_db_name, "", "The output training leveldb name.");
CAFFE2_DEFINE_string(db, "leveldb", "The db type.");
CAFFE2_DEFINE_bool(raw, false,
    "If set, we pre-read the images and store the raw buffer.");
CAFFE2_DEFINE_bool(color, true, "If set, load images in color.");
CAFFE2_DEFINE_int(
    scale,
    256,
    "If caffe2::FLAGS_raw is set, scale the shorter edge to the given value.");
CAFFE2_DEFINE_bool(warp, false, "If warp is set, warp the images to square.");
CAFFE2_DEFINE_int(
    num_threads,
    -1,
    "Number of image parsing and conversion threads.");

namespace caffe2 {

class Converter {
 public:
  explicit Converter() {
    data_ = protos_.add_protos();
    label_ = protos_.add_protos();
    if (caffe2::FLAGS_raw) {
      data_->set_data_type(TensorProto::BYTE);
      data_->add_dims(0);
      data_->add_dims(0);
      if (caffe2::FLAGS_color) {
        data_->add_dims(3);
      }
    } else {
      data_->set_data_type(TensorProto::STRING);
      data_->add_dims(1);
      data_->add_string_data("");
    }
    label_->set_data_type(TensorProto::INT32);
    label_->add_dims(1);
    label_->add_int32_data(0);
  }

  ~Converter() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void queue(const std::pair<std::string, int>& pair) {
    in_.push(pair);
  }

  void start() {
    thread_ = std::thread(&Converter::run, this);
  }

  std::string get() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (out_.empty()) {
      cv_.wait(lock);
    }

    auto value = out_.front();
    out_.pop();
    cv_.notify_one();
    return value;
  }

  void run() {
    const auto& input_folder = caffe2::FLAGS_input_folder;
    std::unique_lock<std::mutex> lock(mutex_);
    std::string value;
    while (!in_.empty()) {
      auto pair = in_.front();
      in_.pop();
      lock.unlock();

      label_->set_int32_data(0, pair.second);

      // Add raw file contents to DB if !raw
      if (!caffe2::FLAGS_raw) {
        std::ifstream image_file_stream(input_folder + pair.first);
        if (!image_file_stream) {
          LOG(ERROR) << "Cannot open " << input_folder << pair.first
                     << ". Skipping.";
        } else {
          data_->mutable_string_data(0)->assign(
              std::istreambuf_iterator<char>(image_file_stream),
              std::istreambuf_iterator<char>());
        }
      } else {
        // Load image
        cv::Mat img = cv::imread(
            input_folder + pair.first,
            caffe2::FLAGS_color ? CV_LOAD_IMAGE_COLOR
                                : CV_LOAD_IMAGE_GRAYSCALE);

        // Resize image
        cv::Mat resized_img;
        int scaled_width, scaled_height;
        if (caffe2::FLAGS_warp) {
          scaled_width = caffe2::FLAGS_scale;
          scaled_height = caffe2::FLAGS_scale;
        } else if (img.rows > img.cols) {
          scaled_width = caffe2::FLAGS_scale;
          scaled_height =
              static_cast<float>(img.rows) * caffe2::FLAGS_scale / img.cols;
        } else {
          scaled_height = caffe2::FLAGS_scale;
          scaled_width =
              static_cast<float>(img.cols) * caffe2::FLAGS_scale / img.rows;
        }
        cv::resize(
            img,
            resized_img,
            cv::Size(scaled_width, scaled_height),
            0,
            0,
            cv::INTER_LINEAR);
        data_->set_dims(0, scaled_height);
        data_->set_dims(1, scaled_width);

        // Assert we don't have to deal with alignment
        DCHECK(resized_img.isContinuous());
        auto nbytes = resized_img.total() * resized_img.elemSize();
        data_->set_byte_data(resized_img.ptr(), nbytes);
      }

      protos_.SerializeToString(&value);

      // Add serialized proto to out queue or wait if it is not empty
      lock.lock();
      while (!out_.empty()) {
        cv_.wait(lock);
      }
      out_.push(value);
      cv_.notify_one();
    }
  }

 protected:
  TensorProtos protos_;
  TensorProto* data_;
  TensorProto* label_;
  std::queue<std::pair<std::string, int>> in_;
  std::queue<std::string> out_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
};

void ConvertImageDataset(
    const string& input_folder,
    const string& list_filename,
    const string& output_db_name,
    const bool /*shuffle*/) {
  std::ifstream list_file(list_filename);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int file_label;
  while (list_file >> filename >> file_label) {
    lines.push_back(std::make_pair(filename, file_label));
  }

  if (caffe2::FLAGS_shuffle) {
    LOG(INFO) << "Shuffling data";
    std::shuffle(lines.begin(), lines.end(), std::default_random_engine(1701));
  }

  auto num_threads = caffe2::FLAGS_num_threads;
  if (num_threads < 1) {
    num_threads = std::thread::hardware_concurrency();
  }

  LOG(INFO) << "Processing " << lines.size() << " images...";
  LOG(INFO) << "Opening DB " << output_db_name;

  auto db = db::CreateDB(caffe2::FLAGS_db, output_db_name, db::NEW);
  auto transaction = db->NewTransaction();

  LOG(INFO) << "Using " << num_threads << " processing threads...";
  std::vector<Converter> converters(num_threads);

  // Queue entries across converters
  for (auto i = 0; i < lines.size(); i++) {
    converters[i % converters.size()].queue(lines[i]);
  }

  // Start all converters
  for (auto& converter : converters) {
    converter.start();
  }

  constexpr auto key_max_length = 256;
  char key_cstr[key_max_length];
  string value;
  int count = 0;
  for (auto i = 0; i < lines.size(); i++) {
    // Get serialized proto for this entry
    auto value = converters[i % converters.size()].get();

    // Synthesize key for this entry
    auto key_len = snprintf(
        key_cstr, sizeof(key_cstr), "%08d_%s", i, lines[i].first.c_str());
    DCHECK_LE(key_len, sizeof(key_cstr));

    // Put in db
    transaction->Put(string(key_cstr), value);

    if (++count % 1000 == 0) {
      // Commit the current writes.
      transaction->Commit();
      LOG(INFO) << "Processed " << count << " files.";
    }
  }

  // Commit final transaction
  transaction->Commit();
  LOG(INFO) << "Processed " << count << " files.";
}

}  // namespace caffe2


int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ConvertImageDataset(
      caffe2::FLAGS_input_folder, caffe2::FLAGS_list_file,
      caffe2::FLAGS_output_db_name, caffe2::FLAGS_shuffle);
  return 0;
}

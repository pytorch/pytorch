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

#include <opencv2/opencv.hpp>
#include <fstream>

#include "caffe2/core/common.h"
#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

C10_DEFINE_bool(color, true, "If set, load images in color.");
C10_DEFINE_string(input_images, "", "Comma separated images");
C10_DEFINE_string(input_image_file, "", "The file containing imput images");
C10_DEFINE_string(output_tensor, "", "The output tensor file in NCHW");
C10_DEFINE_int(scale, 256, "Scale the shorter edge to the given value.");
C10_DEFINE_bool(text_output, false, "Write the output in text format.");
C10_DEFINE_bool(warp, false, "If warp is set, warp the images to square.");
C10_DEFINE_string(
    preprocess,
    "",
    "Options to specify the preprocess routines. The available options are "
    "subtract128, normalize, mean, std, bgrtorgb. If multiple steps are provided, they "
    "are separated by comma (,) in sequence.");

namespace caffe2 {

cv::Mat resizeImage(cv::Mat& img) {
  cv::Mat resized_img;
  int scaled_width, scaled_height;
  if (c10::FLAGS_warp) {
    scaled_width = c10::FLAGS_scale;
    scaled_height = c10::FLAGS_scale;
  } else if (img.rows > img.cols) {
    scaled_width = c10::FLAGS_scale;
    scaled_height = static_cast<float>(img.rows) * c10::FLAGS_scale / img.cols;
  } else {
    scaled_height = c10::FLAGS_scale;
    scaled_width = static_cast<float>(img.cols) * c10::FLAGS_scale / img.rows;
  }
  cv::resize(
      img,
      resized_img,
      cv::Size(scaled_width, scaled_height),
      0,
      0,
      cv::INTER_LINEAR);
  return resized_img;
}

cv::Mat cropToSquare(cv::Mat& img) {
  // Crop image to square
  if (img.rows != img.cols) {
    cv::Mat cropped_img;
    int size = img.rows > img.cols ? img.cols : img.rows;
    cv::Rect roi;
    roi.x = int((img.cols - size) / 2);
    roi.y = int((img.rows - size) / 2);
    roi.width = size;
    roi.height = size;

    cropped_img = img(roi);
    return cropped_img;
  } else {
    return img;
  }
}

std::vector<float> convertToVector(cv::Mat& img) {
  std::vector<float> normalize(3, 1);
  std::vector<float> mean(3, 0);
  std::vector<float> std(3, 1);
  bool bgrtorgb = false;
  assert(img.cols == c10::FLAGS_scale);
  assert(img.rows == c10::FLAGS_scale);
  vector<string> steps = caffe2::split(',', c10::FLAGS_preprocess);
  for (int i = 0; i < steps.size(); i++) {
    auto step = steps[i];
    if (step == "subtract128") {
      mean = {128, 128, 128};
      std = {1, 1, 1};
      normalize = {1, 1, 1};
    } else if (step == "normalize") {
      normalize = {255, 255, 255};
    } else if (step == "mean") {
      mean = {0.406, 0.456, 0.485};
    } else if (step == "std") {
      std = {0.225, 0.224, 0.229};
    } else if (step == "bgrtorgb") {
      bgrtorgb = true;
    } else {
      CAFFE_ENFORCE(
          false,
          "Unsupported preprocess step. The supported steps are: subtract128, "
          "normalize,mean, std, swaprb.");
    }
  }

  int C = c10::FLAGS_color ? 3 : 1;
  int total_size = C * c10::FLAGS_scale * c10::FLAGS_scale;
  std::vector<float> values(total_size);
  if (C == 1) {
    cv::MatIterator_<uchar> it, end;
    int idx = 0;
    for (it = img.begin<uchar>(), end = img.end<uchar>(); it != end; ++it) {
      values[idx++] = (*it / normalize[0] - mean[0]) / std[0];
    }
  } else {
    int i = 0;
    cv::MatIterator_<cv::Vec3b> it, end;
    int b = bgrtorgb ? 2 : 0;
    int g = 1;
    int r = bgrtorgb ? 0 : 2;
    for (it = img.begin<cv::Vec3b>(), end = img.end<cv::Vec3b>(); it != end;
         ++it, i++) {
      values[i] = (((*it)[b] / normalize[0] - mean[0]) / std[0]);
      int offset = c10::FLAGS_scale * c10::FLAGS_scale + i;
      values[offset] = (((*it)[g] / normalize[1] - mean[1]) / std[1]);
      offset = c10::FLAGS_scale * c10::FLAGS_scale + offset;
      values[offset] = (((*it)[r] / normalize[2] - mean[2]) / std[2]);
    }
  }
  return values;
}

std::vector<float> convertOneImage(std::string& filename) {
  assert(filename[0] != '~');

  std::cout << "Converting " << filename << std::endl;
  // Load image
  cv::Mat img = cv::imread(
      filename,
      c10::FLAGS_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat crop = cropToSquare(img);

  // Resize image
  cv::Mat resized_img = resizeImage(crop);
  // Assert we don't have to deal with alignment
  DCHECK(resized_img.isContinuous());
  assert(resized_img.rows == resized_img.cols);
  assert(resized_img.rows == c10::FLAGS_scale);
  std::vector<float> one_image_values = convertToVector(resized_img);
  return one_image_values;
}

void convertImages() {
  vector<string> file_names;
  if (c10::FLAGS_input_images != "") {
    file_names = caffe2::split(',', c10::FLAGS_input_images);
  } else if (c10::FLAGS_input_image_file != "") {
    std::ifstream infile(c10::FLAGS_input_image_file);
    std::string line;
    while (std::getline(infile, line)) {
      vector<string> file_name = caffe2::split(',', line);
      string name;
      if (file_name.size() == 3) {
        name = file_name[2];
      } else {
        name = line;
      }
      file_names.push_back(name);
    }
  } else {
    assert(false);
  }
  std::vector<std::vector<float>> values;
  int C = c10::FLAGS_color ? 3 : 1;
  for (int i = 0; i < file_names.size(); i++) {
    std::vector<float> one_image_values = convertOneImage(file_names[i]);
    values.push_back(one_image_values);
  }

  TensorProtos protos;
  TensorProto* data;
  data = protos.add_protos();
  data->set_data_type(TensorProto::FLOAT);
  data->add_dims(values.size());
  data->add_dims(C);
  data->add_dims(c10::FLAGS_scale);
  data->add_dims(c10::FLAGS_scale);

  for (int i = 0; i < values.size(); i++) {
    assert(values[i].size() == C * c10::FLAGS_scale * c10::FLAGS_scale);
    for (int j = 0; j < values[i].size(); j++) {
      data->add_float_data(values[i][j]);
    }
  }
  if (c10::FLAGS_text_output) {
    caffe2::WriteProtoToTextFile(protos, c10::FLAGS_output_tensor);
  } else {
    caffe2::WriteProtoToBinaryFile(protos, c10::FLAGS_output_tensor);
  }
}

} // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::convertImages();
  return 0;
}

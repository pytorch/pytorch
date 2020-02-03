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
#include <cmath>
#include <fstream>

#include "caffe2/core/common.h"
#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"


C10_DEFINE_int(
    batch_size,
    -1,
    "Specify the batch size of the input. The number of items in the "
    "input needs to be multiples of the batch size. If the batch size "
    "is less than 0, all inputs are in one batch.")
C10_DEFINE_bool(color, true, "If set, load images in color.");
C10_DEFINE_string(
    crop,
    "-1,-1",
    "The center cropped hight and width. If the value is less than zero, "
    "it is not cropped.");
C10_DEFINE_string(input_images, "", "Comma separated images");
C10_DEFINE_string(input_image_file, "", "The file containing imput images");
C10_DEFINE_string(input_text_file, "", "the text file to be written to blobs");
C10_DEFINE_string(
    output_tensor,
    "",
    "The output tensor file in NCHW for input images");
C10_DEFINE_string(
    output_text_tensor,
    "",
    "The output tensor file for the text input specified in input_text_file");
C10_DEFINE_string(
    preprocess,
    "",
    "Options to specify the preprocess routines. The available options are "
    "subtract128, normalize, mean, std, bgrtorgb. If multiple steps are provided, they "
    "are separated by comma (,) in sequence.");
C10_DEFINE_string(
    report_time,
    "",
    "Report the conversion stage time to screen. "
    "The format of the string is <type>|<identifier>. "
    "The valid type is 'json'. "
    "The valid identifier is nothing or an identifier that prefix every line");
C10_DEFINE_string(
    scale,
    "-1,-1",
    "Scale the images to be within the min,max box. The shorter edge is "
    "min pixels. But if the other edge is more than the max pixels, the "
    "other edge and scaled to max pixels (and the shorter edge can be less "
    "than the min pixels");
C10_DEFINE_bool(text_output, false, "Write the output in text format.");
C10_DEFINE_bool(warp, false, "If warp is set, warp the images to square.");

namespace caffe2 {

void reportTime(
    std::string type,
    double ts,
    std::string metric,
    std::string unit) {
  if (FLAGS_report_time == "") {
    return;
  }
  vector<string> s = caffe2::split('|', FLAGS_report_time);
  assert(s[0] == "json");
  std::string identifier = "";
  if (s.size() > 1) {
    identifier = s[1];
  }
  std::cout << identifier << "{\"type\": \"" << type << "\", \"value\": " << ts
            << ", \"metric\": \"" << metric << "\", \"unit\": \"" << unit
            << "\"}" << std::endl;
}

void splitSizes(const std::string& arg, int* ptr0, int* ptr1) {
  vector<string> sizes = caffe2::split(',', arg);
  if (sizes.size() == 2) {
    *ptr0 = std::stoi(sizes[0]);
    *ptr1 = std::stoi(sizes[1]);
  } else if (sizes.size() == 1) {
    *ptr0 = std::stoi(sizes[0]);
    *ptr1 = std::stoi(sizes[0]);
  } else {
    assert(false);
  }
}


cv::Mat resizeImage(cv::Mat& img) {
  int min_size, max_size;
  splitSizes(FLAGS_scale, &min_size, &max_size);
  if ((min_size <= 0) && (max_size <= 0)) {
    return img;
  }
  if (max_size < 0) {
    max_size = INT_MAX;
  }
  assert(min_size <= max_size);

  int im_min_size = img.rows > img.cols ? img.cols : img.rows;
  int im_max_size = img.rows > img.cols ? img.rows : img.cols;

  double im_scale = 1.0 * min_size / im_min_size;
  if (im_scale * im_max_size > max_size) {
    im_scale = 1.0 * max_size / im_max_size;
  }
  int scaled_width = int(round(img.cols * im_scale));
  int scaled_height = int(round(img.rows * im_scale));
  assert((scaled_width <= max_size) && (scaled_height <= max_size));
  if ((scaled_width < min_size) || (scaled_height < min_size)) {
    assert((scaled_width == max_size) || (scaled_height == max_size));
  } else {
    assert((scaled_width == min_size) || (scaled_height == min_size));
  }
  cv::Mat resized_img;
  cv::resize(
      img,
      resized_img,
      cv::Size(),
      im_scale,
      im_scale,
      cv::INTER_LINEAR);
  return resized_img;
}

cv::Mat cropToRec(cv::Mat& img, int* height_ptr, int* width_ptr) {
  int height = *height_ptr;
  int width = *width_ptr;
  if ((height > 0) && (width > 0) &&
      ((img.rows != height) || (img.cols != width))) {
    cv::Mat cropped_img, cimg;
    cv::Rect roi;
    roi.x = int((img.cols - width) / 2);
    roi.y = int((img.rows - height) / 2);
    roi.x = roi.x < 0 ? 0 : roi.x;
    roi.y = roi.y < 0 ? 0 : roi.y;
    width = width > img.cols ? img.cols : width;
    height = height > img.rows ? img.rows : height;
    roi.width = width;
    roi.height = height;
    assert(
        0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= img.cols &&
        0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= img.rows);
    cropped_img = img(roi);
    // Make the image in continuous space in memory
    cimg = cropped_img.clone();
    *height_ptr = height;
    *width_ptr = width;
    return cimg;
  } else {
    return img;
  }
}

std::vector<float> convertToVector(cv::Mat& img) {
  std::vector<float> normalize(3, 1);
  std::vector<float> mean(3, 0);
  std::vector<float> std(3, 1);
  bool bgrtorgb = false;
  int size = img.cols * img.rows;
  vector<string> steps = caffe2::split(',', FLAGS_preprocess);
  for (int i = 0; i < steps.size(); i++) {
    auto step = steps[i];
    if (step == "subtract128") {
      mean = {128, 128, 128};
      std = {1, 1, 1};
      normalize = {1, 1, 1};
    } else if (step == "normalize") {
      normalize = {255, 255, 255};
    } else if (step == "mean") {
      mean = {0.406f, 0.456f, 0.485f};
    } else if (step == "std") {
      std = {0.225f, 0.224f, 0.229f};
    } else if (step == "bgrtorgb") {
      bgrtorgb = true;
    } else {
      CAFFE_ENFORCE(
          false,
          "Unsupported preprocess step. The supported steps are: subtract128, "
          "normalize,mean, std, swaprb.");
    }
  }

  int C = FLAGS_color ? 3 : 1;
  int total_size = C * size;
  std::vector<float> values(total_size);
  if (C == 1) {
    cv::MatIterator_<float> it, end;
    int idx = 0;
    for (it = img.begin<float>(), end = img.end<float>(); it != end; ++it) {
      values[idx++] = (*it / normalize[0] - mean[0]) / std[0];
    }
  } else {
    int i = 0;
    cv::MatIterator_<cv::Vec3f> it, end;
    int b = bgrtorgb ? 2 : 0;
    int g = 1;
    int r = bgrtorgb ? 0 : 2;
    for (it = img.begin<cv::Vec3f>(), end = img.end<cv::Vec3f>(); it != end;
         ++it, i++) {
      values[i] = (((*it)[b] / normalize[0] - mean[0]) / std[0]);
      int offset = size + i;
      values[offset] = (((*it)[g] / normalize[1] - mean[1]) / std[1]);
      offset = size + offset;
      values[offset] = (((*it)[r] / normalize[2] - mean[2]) / std[2]);
    }
  }
  return values;
}

std::vector<float> convertOneImage(
    std::string& filename,
    int* height_ptr,
    int* width_ptr) {
  assert(filename[0] != '~');

  std::cout << "Converting " << filename << std::endl;

  // Load image
  cv::Mat img_uint8 = cv::imread(
#if CV_MAJOR_VERSION <= 3
      filename, FLAGS_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
#else
      filename, FLAGS_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
#endif
  caffe2::Timer timer;
  timer.Start();
  cv::Mat img;
  // Convert image to floating point values
  img_uint8.convertTo(img, CV_32F);
  // Resize image
  cv::Mat resized_img = resizeImage(img);

  int height, width;
  splitSizes(FLAGS_crop, &height, &width);
  if ((height <= 0) || (width <= 0)) {
    height = resized_img.rows;
    width = resized_img.cols;
  }
  cv::Mat crop = cropToRec(resized_img, &height, &width);

  // Assert we don't have to deal with alignment
  DCHECK(crop.isContinuous());
  assert(crop.rows == height);
  assert(crop.cols == width);
  std::vector<float> one_image_values = convertToVector(crop);
  *height_ptr = height;
  *width_ptr = width;
  double ts = timer.MicroSeconds();
  reportTime("image_preprocess", ts, "convert", "us");
  return one_image_values;
}

int getBatchSize(int num_items) {
  int batch_size = FLAGS_batch_size;
  if (batch_size < 0) {
    batch_size = num_items;
  } else {
    assert(num_items % batch_size == 0);
  }
  return batch_size;
}

void writeValues(
    std::vector<std::vector<std::vector<float>>>& values,
    std::vector<std::vector<int>>& dims,
    std::string output_file) {

  caffe2::Timer timer;
  timer.Start();

  assert(dims.size() == values.size());
  int num_batches = dims.size();

  TensorProtos protos;
  for (int k = 0; k < num_batches; k++) {
    TensorProto* data;
    data = protos.add_protos();
    data->set_data_type(TensorProto::FLOAT);
    auto one_dim = dims[k];
    for (int dim : one_dim) {
      data->add_dims(dim);
    }
    int batch_size = one_dim[0];
    long long int entry_size = 1;
    for (int i = 1; i < one_dim.size(); i++) {
      entry_size *= one_dim[i];
    }

    // Not optimized
    for (int i = 0; i < batch_size; i++) {
      assert(values[k][i].size() == entry_size);
      for (int j = 0; j < values[k][i].size(); j++) {
        data->add_float_data(values[k][i][j]);
      }
    }
  }
  double ts = timer.MicroSeconds();
  reportTime("preprocess", ts, "data_pack", "us");

  if (FLAGS_text_output) {
    caffe2::WriteProtoToTextFile(protos, output_file);
  } else {
    caffe2::WriteProtoToBinaryFile(protos, output_file);
  }
}

void convertImages() {
  vector<string> file_names;
  if (FLAGS_input_images != "") {
    file_names = caffe2::split(',', FLAGS_input_images);
  } else if (FLAGS_input_image_file != "") {
    std::ifstream infile(FLAGS_input_image_file);
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
    return;
  }
  int batch_size = getBatchSize(file_names.size());
  int num_batches = file_names.size() / batch_size;
  assert(file_names.size() == batch_size * num_batches);
  std::vector<std::vector<std::vector<float>>> values;
  std::vector<std::vector<int>> dims;
  int C = FLAGS_color ? 3 : 1;
  for (int k = 0; k < num_batches; k++) {
    std::vector<std::vector<float>> one_value;
    int height = -1;
    int width = -1;
    for (int i = 0; i < batch_size; i++) {
      int idx = k * batch_size + i;
      int one_height, one_width;
      std::vector<float> one_image_values =
          convertOneImage(file_names[idx], &one_height, &one_width);
      if (height < 0 && width < 0) {
        height = one_height;
        width = one_width;
      } else {
        assert(height == one_height);
        assert(width == one_width);
      }
      one_value.push_back(one_image_values);
    }
    vector<int> one_dim = {batch_size, C, height, width};
    dims.push_back(one_dim);
    values.push_back(one_value);
  }
  writeValues(values, dims, FLAGS_output_tensor);
}

template <class TYPE>
vector<TYPE> splitString(std::string& line) {
  vector<string> vector_str = caffe2::split(',', line);
  vector<TYPE> vector_int;
  for (string str : vector_str) {
    vector_int.push_back((TYPE)std::stod(str));
  }
  return vector_int;
}

/* Convert the values in a json file to blobs
   The format of the json file should be:
   <number of items>,  <dim2>.... (dimensions of items)
   <entry>, <entry>, <entry>... (all entries in one item)
   <entry>, <entry>, <entry>...
   ....
*/
void convertValues() {
  if (FLAGS_input_text_file == "") {
    return;
  }
  std::ifstream infile(FLAGS_input_text_file);
  std::string line;
  std::getline(infile, line);
  vector<int> file_dims = splitString <int>(line);
  assert(file_dims.size() >= 2);

  int num_items = file_dims[0];
  int batch_size = getBatchSize(num_items);
  int num_batches = num_items / batch_size;
  assert(num_items == batch_size * num_batches);
  vector<string> lines;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }
  assert(lines.size() == num_items);
  std::vector<std::vector<std::vector<float>>> values;
  std::vector<std::vector<int>> dims;
  for (int i = 0; i < num_batches; i++) {
    std::vector<std::vector<float>> one_value;
    int num = -1;
    for (int j = 0; j < batch_size; j++) {
      int idx = i * batch_size + j;
      std::string line = lines[idx];
      vector<float> item = splitString<float>(line);
      if (num < 0) {
        num = item.size();
      } else {
        assert(num == item.size());
      }
      one_value.push_back(item);
    }
    vector<int> batch_dims = file_dims;
    batch_dims[0] = batch_size;
    dims.push_back(batch_dims);
    values.push_back(one_value);
  }

  writeValues(values, dims, FLAGS_output_text_tensor);
}

} // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::convertImages();
  caffe2::convertValues();
  return 0;
}

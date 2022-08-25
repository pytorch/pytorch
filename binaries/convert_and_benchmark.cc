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
#include "caffe2/utils/bench_utils.h"

#include "binaries/benchmark_args.h"
#include "binaries/benchmark_helper.h"

#include <observers/net_observer_reporter_print.h>
#include <observers/observer_config.h>
#include <observers/perf_observer.h>


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
C10_DEFINE_string(input_image_files, "", "Files containing imput images");
C10_DEFINE_string(input_text_files, "", "Text files to be written to blobs");
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

TensorProtos writeValues(
    std::vector<std::vector<std::vector<float>>>& values,
    std::vector<std::vector<int>>& dims) {

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

  return protos;
}

TensorProtos convertImages(std::string& image_file) {
  vector<string> file_names;
  if (image_file != "") {
    std::ifstream infile(image_file);
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
    TensorProtos proto;
    return proto;
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
  return writeValues(values, dims);
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
TensorProtos convertValues(std::string& file_name) {
  if (file_name == "") {
    TensorProtos proto;
    return proto;
  }
  std::ifstream infile(file_name);
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

  return writeValues(values, dims);
}

} // namespace caffe2

void observerConfig() {
  caffe2::ClearGlobalNetObservers();
  caffe2::AddGlobalNetObserverCreator([](caffe2::NetBase* subject) {
    return std::make_unique<caffe2::PerfNetObserver>(subject);
  });
  caffe2::ObserverConfig::setReporter(
      std::make_unique<caffe2::NetObserverReporterPrint>());
}

bool backendCudaSet(const string& backend) {
  bool run_on_gpu = false;
  if (backend == "cuda") {
#ifdef __CUDA_ARCH__
    if (caffe2::HasCudaGPU()) {
      run_on_gpu = true;
    } else {
      CAFFE_THROW("NO GPU support on this host machine");
    }
#else
    CAFFE_THROW("NO GPU support");
#endif
  }
  return run_on_gpu;
}

void setOperatorEngine(caffe2::NetDef* net_def, const string& backend) {
  if (backend != "builtin") {
    string engine;
    if( backend == "nnpack" ) {
      engine = "NNPACK";
    } else if ( backend == "eigen" ) {
      engine = "EIGEN";
    } else if ( backend == "mkl" ) {
      engine = "MKLDNN";
    } else if ( backend == "cuda" ) {
      engine = "CUDA";
    } else if ( backend == "dnnlowp" ) {
      engine = "DNNLOWP";
    } else if ( backend == "dnnlowp_acc16" ) {
      engine = "DNNLOWP_ACC16";
    } else if ( backend == "default" ) {
      engine = "";
    } else {
      engine = "NONE";
    }
    CAFFE_ENFORCE(engine != "NONE", "Backend is not supported");
    for (int i = 0; i < net_def->op_size(); i++) {
      caffe2::OperatorDef* op_def = net_def->mutable_op(i);
      op_def->set_engine(engine);
    }
  }
}

void fillInputBlob(
    shared_ptr<caffe2::Workspace> workspace,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    int iteration) {
  if (tensor_protos_map.empty()) {
    return;
  }
  static caffe2::TensorDeserializer deserializer;
  for (auto& tensor_kv : tensor_protos_map) {
    caffe2::Blob* blob = workspace->GetBlob(tensor_kv.first);
    if (blob == nullptr) {
      blob = workspace->CreateBlob(tensor_kv.first);
    }
    // todo: support gpu and make this function a template
    int protos_size = tensor_kv.second.protos_size();
    if (protos_size == 1 && iteration > 0) {
      // Do not override the input data if there is only one input data,
      // since it will clear all caches. Rely on wipe_cache to
      // clear caches
      continue;
    }
    caffe2::TensorProto* tensor_proto =
        tensor_kv.second.mutable_protos(iteration % protos_size);
    BlobSetTensor(blob, deserializer.Deserialize(*tensor_proto));
    // todo: for other types
  }
}

void writeOutput(
    shared_ptr<caffe2::Workspace> workspace,
    const bool run_on_gpu,
    const string& output,
    const string& output_folder,
    const bool text_output,
    const int index,
    const int num_blobs) {
  if (output.size() == 0) {
    return;
  }
  string output_prefix = output_folder.size() ? output_folder + "/" : "";
  vector<string> output_names = caffe2::split(',', output);
  if (output == "*") {
    output_names = workspace->Blobs();
  }
  for (const string& name : output_names) {
    CAFFE_ENFORCE(
        workspace->HasBlob(name),
        "You requested a non-existing blob: ",
        name);
    if (text_output) {
      if (run_on_gpu) {
#ifdef __CUDA_ARCH__
        writeTextOutput<caffe2::CUDAContext, caffe2::TensorCUDA>(
            workspace->GetBlob(name)->GetMutable<caffe2::TensorCUDA>(),
            output_prefix,
            name,
            index,
            num_blobs);
#else
        CAFFE_THROW("Not support GPU.");
#endif
      } else {
        writeTextOutput<caffe2::CPUContext, caffe2::TensorCPU>(
            BlobGetMutableTensor(workspace->GetBlob(name), caffe2::CPU),
            output_prefix,
            name,
            index,
            num_blobs);
      }
    } else {
      // Do not support multiple entries per blob.
      CAFFE_ENFORCE(
          index == 0,
          "Binary file only support one output.");
      string serialized = SerializeBlob(*workspace->GetBlob(name), name);
      string output_filename = output_prefix + name;
      caffe2::WriteStringToFile(serialized, output_filename.c_str());
    }
  }
}

void runNetwork(
    shared_ptr<caffe2::Workspace> workspace,
    caffe2::NetDef& net_def,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    const bool wipe_cache,
    const bool run_individual,
    const bool run_on_gpu,
    const bool text_output,
    const int warmup,
    const int iter,
    const int num_blobs,
    const int sleep_before_run,
    const int sleep_between_iteration,
    const int sleep_between_net_and_operator,
    const std::string& output,
    const std::string& output_folder) {

  if (!net_def.has_name()) {
    net_def.set_name("benchmark");
  }

  caffe2::NetBase* net = workspace->CreateNet(net_def);
  TORCH_CHECK_NOTNULL(net);

  LOG(INFO) << "Starting benchmark.";
  caffe2::ObserverConfig::initSampleRate(1, 1, 1, run_individual, warmup);
  LOG(INFO) << "Running warmup runs.";
  for (int i = 0; i < warmup; ++i) {
    fillInputBlob(workspace, tensor_protos_map, i);
    CAFFE_ENFORCE(net->Run(), "Warmup run ", i, " has failed.");
  }

  if (wipe_cache) {
    caffe2::wipe_cache();
  }
  if (sleep_before_run > 0) {
    std::this_thread::sleep_for(std::chrono::seconds(sleep_before_run));
  }
  LOG(INFO) << "Main runs.";
  CAFFE_ENFORCE(
      iter >= 0,
      "Number of main runs should be non negative, provided ",
      iter,
      ".");
  LOG(INFO) << "net runs.";
  for (int i = 0; i < iter; ++i) {
    caffe2::ObserverConfig::initSampleRate(1, 1, 1, 0, warmup);
    fillInputBlob(workspace, tensor_protos_map, i);
    if (wipe_cache) {
      caffe2::wipe_cache();
    }
    CAFFE_ENFORCE(net->Run(), "Main run ", i, " has failed.");
    // Write the output for the first num_blobs times
    writeOutput(
        workspace,
        run_on_gpu,
        output,
        output_folder,
        text_output,
        i,
        num_blobs);
    if (wipe_cache) {
      caffe2::wipe_cache();
    }
    if (sleep_between_iteration > 0) {
      std::this_thread::sleep_for(
          std::chrono::seconds(sleep_between_iteration));
    }
  }
  if (run_individual) {
    LOG(INFO) << "operator runs.";
    if (sleep_between_net_and_operator > 0) {
      std::this_thread::sleep_for(
          std::chrono::seconds(sleep_between_net_and_operator));
    }
    for (int i = 0; i < iter; ++i) {
      caffe2::ObserverConfig::initSampleRate(1, 1, 1, 1, warmup);
      fillInputBlob(workspace, tensor_protos_map, i);
      CAFFE_ENFORCE(net->Run(), "Main run ", i, " with operator has failed.");
      if (wipe_cache) {
        caffe2::wipe_cache();
      }
      if (sleep_between_iteration > 0) {
        std::this_thread::sleep_for(
            std::chrono::seconds(sleep_between_iteration));
      }
    }
  }
}

int benchmark(
    int argc,
    char* argv[],
    const string& FLAGS_backend,
    const string& FLAGS_init_net,
    const string& FLAGS_input_dims,
    int FLAGS_iter,
    const string& FLAGS_net,
    const string& FLAGS_output,
    const string& FLAGS_output_folder,
    bool FLAGS_run_individual,
    int FLAGS_sleep_before_run,
    int FLAGS_sleep_between_iteration,
    int FLAGS_sleep_between_net_and_operator,
    bool FLAGS_text_output,
    int FLAGS_warmup,
    bool FLAGS_wipe_cache) {
  // Check arguments to be correct
  {
    // Need to check whether file exists, as the file reader does not assert if
    // file does not exist
    std::ifstream net_file(FLAGS_net);
    CAFFE_ENFORCE(net_file.good());
    net_file.close();

    std::ifstream init_net_file(FLAGS_init_net);
    CAFFE_ENFORCE(init_net_file.good());
    init_net_file.close();
  }

  observerConfig();
  caffe2::ShowLogInfoToStderr();

  auto workspace = std::make_shared<caffe2::Workspace>(new caffe2::Workspace());
  bool run_on_gpu = backendCudaSet(FLAGS_backend);
  // Run initialization network.
  caffe2::NetDef init_net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net_def));
  setOperatorEngine(&init_net_def, FLAGS_backend);
  CAFFE_ENFORCE(workspace->RunNetOnce(init_net_def));

  // Run main network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_net, &net_def));
  setOperatorEngine(&net_def, FLAGS_backend);

  map<string, caffe2::TensorProtos> tensor_protos_map;

  int num_blobs;
  vector<string> images = caffe2::split(';', FLAGS_input_image_files);
  for (int i = 0; i < images.size(); ++i) {
    vector<string> mapping = caffe2::split(',', images[i]);
    caffe2::TensorProtos proto_images = caffe2::convertImages(mapping[1]);
    workspace->CreateBlob(mapping[0]);
    tensor_protos_map.insert(std::make_pair(mapping[0], proto_images));
    num_blobs = proto_images.protos_size();
  }

  vector<string> values = caffe2::split(';', FLAGS_input_text_files);
  for (int i = 0; i < values.size(); ++i) {
    vector<string> mapping = caffe2::split(',', values[i]);
    caffe2::TensorProtos proto_values = caffe2::convertValues(mapping[1]);
    workspace->CreateBlob(mapping[0]);
    tensor_protos_map.insert(std::make_pair(mapping[0], proto_values));
    num_blobs = proto_values.protos_size();
  }

  runNetwork(
      workspace,
      net_def,
      tensor_protos_map,
      FLAGS_wipe_cache,
      FLAGS_run_individual,
      run_on_gpu,
      FLAGS_text_output,
      FLAGS_warmup,
      FLAGS_iter,
      num_blobs,
      FLAGS_sleep_before_run,
      FLAGS_sleep_between_iteration,
      FLAGS_sleep_between_net_and_operator,
      FLAGS_output,
      FLAGS_output_folder);

  return 0;
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  benchmark(
      argc,
      argv,
      FLAGS_backend,
      FLAGS_init_net,
      FLAGS_input_dims,
      FLAGS_iter,
      FLAGS_net,
      FLAGS_output,
      FLAGS_output_folder,
      FLAGS_run_individual,
      FLAGS_sleep_before_run,
      FLAGS_sleep_between_iteration,
      FLAGS_sleep_between_net_and_operator,
      FLAGS_text_output,
      FLAGS_warmup,
      FLAGS_wipe_cache);

  return 0;
}

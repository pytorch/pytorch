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

#include <string>

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

CAFFE2_DEFINE_string(net, "", "The given net to benchmark.");
CAFFE2_DEFINE_string(
    optimized_net,
    "",
    "Output filename for optimized NetDef protobuf.");
CAFFE2_DEFINE_string(
    init_net,
    "",
    "The given net to initialize any parameters.");
CAFFE2_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
CAFFE2_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
CAFFE2_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
CAFFE2_DEFINE_string(input_type, "", "Input type (uint8_t/float)");
CAFFE2_DEFINE_bool(
    try_winograd_fp16,
    true,
    "Consider WINOGRAD_FP16 NNPACK algorithm in auto-tuning.");
CAFFE2_DEFINE_int(
    samples,
    7,
    "The number of samples of warm-up & main iterations to run.");
CAFFE2_DEFINE_int(warmup, 5, "The number of warm-up iterations to run.");
CAFFE2_DEFINE_int(iter, 15, "The number of iterations to run.");

using std::string;
using std::unique_ptr;
using std::vector;

struct InputDescription {
  InputDescription(const string& name, const caffe2::BlobProto& blob)
      : name(name), blob(blob) {}

  InputDescription(const string& name, caffe2::BlobProto&& blob)
      : name(name), blob(blob) {}

  InputDescription(const string& name, const vector<int> shape, bool is_float)
      : name(name), shape(shape), is_float(is_float) {}

  string name;
  /*
   * Protobuf with blob data and metadata.
   * If this value is not initialized, a new blob is created based on shape and
   * is_float values.
   */
  caffe2::BlobProto blob;
  /* Shape of the input (ignored if blob is initialized). */
  vector<int> shape;
  /* If true, input is of float type. Otherwise, it is of uint8_t type. */
  bool is_float;
};

bool benchmark(
    const vector<InputDescription>& inputs,
    const caffe2::NetDef& ref_def,
    const caffe2::NetDef& alt_def,
    const caffe2::NetDef& init_net,
    uint32_t iterations,
    float* millis) {
  unique_ptr<caffe2::Workspace> ref_workspace(new caffe2::Workspace());
  CAFFE_ENFORCE(ref_workspace->RunNetOnce(init_net));

  unique_ptr<caffe2::Workspace> alt_workspace(new caffe2::Workspace());
  CAFFE_ENFORCE(alt_workspace->RunNetOnce(init_net));

  for (const InputDescription& input : inputs) {
    if (false /* input.blob.IsInitialized() */) {
      ref_workspace->CreateBlob(input.name)->Deserialize(input.blob);
      alt_workspace->CreateBlob(input.name)->Deserialize(input.blob);
    } else {
      caffe2::TensorCPU* ref_tensor = ref_workspace->CreateBlob(input.name)
                                          ->GetMutable<caffe2::TensorCPU>();
      caffe2::TensorCPU* alt_tensor = alt_workspace->CreateBlob(input.name)
                                          ->GetMutable<caffe2::TensorCPU>();
      ref_tensor->Resize(input.shape);
      alt_tensor->Resize(input.shape);
      if (input.is_float) {
        ref_tensor->mutable_data<float>();
        alt_tensor->mutable_data<float>();
      } else {
        ref_tensor->mutable_data<uint8_t>();
        alt_tensor->mutable_data<uint8_t>();
      }
    }
  }

  caffe2::NetBase* ref_net = ref_workspace->CreateNet(ref_def);
  caffe2::NetBase* alt_net = alt_workspace->CreateNet(alt_def);

  CAFFE_ENFORCE(alt_net->Run(), "Warmup run for alternative net has failed.");

  vector<float> ref_millis(caffe2::FLAGS_samples);
  vector<float> alt_millis(caffe2::FLAGS_samples);
  for (int n = 0; n < caffe2::FLAGS_samples; n++) {
    {
      /* Reference network */
      for (int i = 0; i < caffe2::FLAGS_warmup; i++) {
        CAFFE_ENFORCE(
            ref_net->Run(), "Warmup run for reference net has failed.");
      }
      caffe2::Timer timer;
      for (int i = 0; i < caffe2::FLAGS_iter; i++) {
        CAFFE_ENFORCE(
            ref_net->Run(), "Main run ", i, " for reference net has failed.");
      }
      ref_millis[n] = timer.MilliSeconds();
    }
    {
      /* Alternative network */
      for (int i = 0; i < caffe2::FLAGS_warmup; i++) {
        CAFFE_ENFORCE(
            alt_net->Run(), "Warmup run for reference net has failed.");
      }
      caffe2::Timer timer;
      for (int i = 0; i < caffe2::FLAGS_iter; i++) {
        CAFFE_ENFORCE(
            alt_net->Run(), "Main run ", i, " for reference net has failed.");
      }
      alt_millis[n] = timer.MilliSeconds();
    }
  }
  std::sort(ref_millis.begin(), ref_millis.end());
  std::sort(alt_millis.begin(), alt_millis.end());
  millis[0] = ref_millis[caffe2::FLAGS_samples / 2] / caffe2::FLAGS_iter;
  millis[1] = alt_millis[caffe2::FLAGS_samples / 2] / caffe2::FLAGS_iter;
  return millis[1] < millis[0];
}

void try_nnpack_convolution(
    const vector<InputDescription>& inputs,
    const caffe2::NetDef& init_net,
    caffe2::NetDef& best_net,
    int conv_index,
    std::string algorithm,
    std::string strategy,
    int shared_buffer) {
  caffe2::NetDef candidate_net(best_net);
  caffe2::OperatorDef* candidate_conv = candidate_net.mutable_op(conv_index);
  CHECK_NOTNULL(candidate_conv);
  caffe2::AddArgument("engine", string("NNPACK"), candidate_conv);
  caffe2::AddArgument("algo", algorithm, candidate_conv);
  caffe2::AddArgument(
      "convolution_transform_strategy", strategy, candidate_conv);
  caffe2::AddArgument("shared_buffer", shared_buffer, candidate_conv);
  float millis[2] = {0.0f, 0.0f};
  if (benchmark(
          inputs,
          best_net,
          candidate_net,
          init_net,
          caffe2::FLAGS_iter,
          millis)) {
    std::cout << "\tImprovement " << std::fixed << std::setprecision(2)
              << millis[0] << " ms -> " << millis[1] << " ms: "
              << "engine = \"NNPACK\", "
              << "algo = \"" << algorithm << "\", "
              << "convolution_transform_strategy = \"" << strategy << "\", "
              << "shared_buffer = " << shared_buffer << std::endl;
    best_net.Clear();
    best_net.CopyFrom(candidate_net);
  }
}

string op_name(const caffe2::OperatorDef& op, int index) {
  if (op.has_name() && op.name().size() != 0) {
    return "\"" + op.name() + "\"";
  } else {
    return "#" + caffe2::to_string(index);
  }
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  /* Validate arguments */
  CAFFE_ENFORCE(
      caffe2::FLAGS_net.size() != 0,
      "Unspecified path to input network protobuf");
  CAFFE_ENFORCE(
      caffe2::FLAGS_optimized_net.size() != 0,
      "Unspecified path to output network protobuf");
  CAFFE_ENFORCE(
      caffe2::FLAGS_init_net.size() != 0,
      "Unspecified path to input initialization protobuf");

  /* Parse input-related options and load data. */
  vector<InputDescription> inputs;
  if (caffe2::FLAGS_input.size()) {
    vector<string> input_names = caffe2::split(',', caffe2::FLAGS_input);
    if (caffe2::FLAGS_input_file.size()) {
      CAFFE_ENFORCE_EQ(
          0,
          caffe2::FLAGS_input_dims.size(),
          "Input file and input dims options are mutually exclusive");
      CAFFE_ENFORCE_EQ(
          0,
          caffe2::FLAGS_input_type.size(),
          "Input file and input type options are mutually exclusive");

      vector<string> input_paths = caffe2::split(',', caffe2::FLAGS_input_file);
      CAFFE_ENFORCE_EQ(
          input_paths.size(),
          input_paths.size(),
          "Input name and file should have the same number.");
      for (size_t i = 0; i < input_paths.size(); i++) {
        caffe2::BlobProto blob_proto;
        CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input_paths[i], &blob_proto));
        inputs.push_back(
            InputDescription(input_names[i], std::move(blob_proto)));
      }
    } else if (
        caffe2::FLAGS_input_dims.size() || caffe2::FLAGS_input_type.size()) {
      CAFFE_ENFORCE_NE(
          0,
          caffe2::FLAGS_input_dims.size(),
          "Input dims must be specified when input files are not specified.");
      CAFFE_ENFORCE_NE(
          0,
          caffe2::FLAGS_input_type.size(),
          "Input types must be specified when input files are not specified.");

      vector<string> input_dims_list =
          caffe2::split(';', caffe2::FLAGS_input_dims);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_dims_list.size(),
          "Input names and input dims should have the same number of items.");
      vector<string> input_type_list =
          caffe2::split(';', caffe2::FLAGS_input_type);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_type_list.size(),
          "Input names and input types should have the same number of items.");
      for (size_t i = 0; i < input_names.size(); ++i) {
        vector<string> input_dims_str = caffe2::split(',', input_dims_list[i]);
        vector<int> input_dims;
        for (const string& s : input_dims_str) {
          input_dims.push_back(caffe2::stoi(s));
        }
        if (input_type_list[i] == "uint8_t") {
          inputs.push_back(InputDescription(input_names[i], input_dims, false));
        } else if (input_type_list[i] == "float") {
          inputs.push_back(InputDescription(input_names[i], input_dims, true));
        } else {
          CAFFE_THROW(
              "Unsupported input type ",
              input_type_list[i],
              " for input ",
              input_names[i]);
        }
      }
    } else {
      CAFFE_THROW(
          "You requested input tensors, but neither input_file nor input_dims + input_type is set.");
    }
  }

  /* Load all parameters into a workspace, so we can lookup their shapes when
   * needed. */
  unique_ptr<caffe2::Workspace> paramWorkspace(new caffe2::Workspace());
  caffe2::NetDef init_net;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(paramWorkspace->RunNetOnce(init_net));

  caffe2::NetDef input_net;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &input_net));
  caffe2::NetDef best_net(input_net);

  for (int op_index = 0; op_index < input_net.op_size(); op_index++) {
    const caffe2::OperatorDef& op = input_net.op(op_index);

    if (op.type() == "Conv") {
      CAFFE_ENFORCE(
          op.input_size() >= 2, "Conv operator must have 2 or 3 inputs");
      caffe2::Blob* kernel_blob = paramWorkspace->GetBlob(op.input(1));
      CAFFE_ENFORCE(
          kernel_blob != nullptr,
          "Weights blob ",
          op.input(1),
          " for Conv operator ",
          op.name(),
          " is not initialized");
      const caffe2::TensorCPU kernel_tensor =
          kernel_blob->Get<caffe2::TensorCPU>();

      if (kernel_tensor.ndim() == 4) {
        /* Weights tensor is 4D -> convolution is 2D */
        caffe2::ArgumentHelper args(op);

        caffe2::StorageOrder order = caffe2::StringToStorageOrder(
            args.GetSingleArgument<string>("order", "NCHW"));
        if (order == caffe2::NCHW) {
          const int output_channels = kernel_tensor.dim32(0);
          const int input_channels = kernel_tensor.dim32(1);

          vector<int> kernel(args.GetRepeatedArgument<int>("kernels"));
          if (args.HasArgument("kernel")) {
            kernel.clear();
            kernel.resize(2, args.GetSingleArgument<int>("kernel", 0));
          } else if (
              args.HasArgument("kernel_h") && args.HasArgument("kernel_w")) {
            kernel.clear();
            kernel.push_back(args.GetSingleArgument<int>("kernel_h", 0));
            kernel.push_back(args.GetSingleArgument<int>("kernel_w", 0));
          }

          vector<int> stride(args.GetRepeatedArgument<int>("strides"));
          if (args.HasArgument("stride")) {
            stride.clear();
            stride.resize(2, args.GetSingleArgument<int>("stride", 0));
          } else if (
              args.HasArgument("stride_h") && args.HasArgument("stride_w")) {
            stride.clear();
            stride.push_back(args.GetSingleArgument<int>("stride_h", 0));
            stride.push_back(args.GetSingleArgument<int>("stride_w", 0));
          }

          vector<int> dilation(args.GetRepeatedArgument<int>("dilation"));
          if (args.HasArgument("dilation")) {
            dilation.clear();
            dilation.resize(2, args.GetSingleArgument<int>("dilation", 0));
          } else if (
              args.HasArgument("dilation_h") &&
              args.HasArgument("dilation_w")) {
            dilation.clear();
            dilation.push_back(args.GetSingleArgument<int>("dilation_h", 0));
            dilation.push_back(args.GetSingleArgument<int>("dilation_w", 0));
          }

          int groups = args.GetSingleArgument<int>("group", 1);

          vector<int> padding(args.GetRepeatedArgument<int>("pads"));
          if (args.HasArgument("pad")) {
            padding.clear();
            padding.resize(4, args.GetSingleArgument<int>("pad", 0));
          } else if (
              args.HasArgument("pad_t") && args.HasArgument("pad_l") &&
              args.HasArgument("pad_b") && args.HasArgument("pad_r")) {
            padding.clear();
            padding.push_back(args.GetSingleArgument<int>("pad_t", 0));
            padding.push_back(args.GetSingleArgument<int>("pad_l", 0));
            padding.push_back(args.GetSingleArgument<int>("pad_b", 0));
            padding.push_back(args.GetSingleArgument<int>("pad_r", 0));
          }

          CAFFE_ENFORCE(
              kernel.size() == 2,
              "Conv operator must explicitly specify kernel argument");
          if (stride.size() == 0) {
            stride.resize(kernel.size(), 1);
          }
          if (padding.size() == 0) {
            padding.resize(kernel.size() * 2, 0);
          }
          if (dilation.size() == 0) {
            dilation.resize(kernel.size(), 1);
          }

          CAFFE_ENFORCE_EQ(
              stride.size(), 2, "2D Conv operator must have 2D stride");
          CAFFE_ENFORCE_EQ(
              dilation.size(), 2, "2D Conv operator must have 2D dilation");
          CAFFE_ENFORCE_EQ(
              padding.size(), 4, "2D Conv operator must have 4D padding");

          std::cout << "Conv operator " << op_name(op, op_index)
                    << " is a candidate for auto-tuning" << std::endl;

          {
            caffe2::NetDef candidate_net(best_net);
            caffe2::OperatorDef* candidate_conv =
                candidate_net.mutable_op(op_index);
            CHECK_NOTNULL(candidate_conv);
            caffe2::AddArgument("engine", string(), candidate_conv);
            caffe2::AddArgument("shared_buffer", 1, candidate_conv);
            float millis[2] = {0.0f, 0.0f};
            if (benchmark(
                    inputs,
                    best_net,
                    candidate_net,
                    init_net,
                    caffe2::FLAGS_iter,
                    millis)) {
              std::cout << "\tImprovement " << std::fixed
                        << std::setprecision(2) << millis[0] << " ms -> "
                        << millis[1] << " ms: "
                        << "engine = \"\", shared_buffer = 1" << std::endl;
              best_net.Clear();
              best_net.CopyFrom(candidate_net);
            }
          }
          if (dilation[0] == 1 && dilation[1] == 1) {
            /* Consider NNPACK */
            try_nnpack_convolution(
                inputs, init_net, best_net, op_index, "AUTO", "PRECOMPUTE", 1);

            if (stride[0] == 1 && stride[1] == 1) {
              /* Consider NNPACK with fast convolution */

              if (kernel[0] == 3 && kernel[1] == 3) {
                try_nnpack_convolution(
                    inputs,
                    init_net,
                    best_net,
                    op_index,
                    "WINOGRAD",
                    "PRECOMPUTE",
                    1);

                if (caffe2::FLAGS_try_winograd_fp16) {
                  try_nnpack_convolution(
                      inputs,
                      init_net,
                      best_net,
                      op_index,
                      "WINOGRAD_FP16",
                      "PRECOMPUTE",
                      1);
                }

                if (input_channels == output_channels &&
                    input_channels == groups) {
                  if (caffe2::CPUOperatorRegistry()->Has(
                      caffe2::OpRegistryKey("Conv", "DEPTHWISE_3x3")))
                  {
                    /* Consider DEPTHWISE_3x3 */
                    caffe2::NetDef candidate_net(best_net);
                    caffe2::OperatorDef* candidate_conv =
                        candidate_net.mutable_op(op_index);
                    CHECK_NOTNULL(candidate_conv);
                    caffe2::AddArgument(
                        "engine", string("DEPTHWISE_3x3"), candidate_conv);
                    float millis[2] = {0.0f, 0.0f};
                    if (benchmark(
                            inputs,
                            best_net,
                            candidate_net,
                            init_net,
                            caffe2::FLAGS_iter,
                            millis)) {
                      std::cout << "\tImprovement " << std::fixed
                                << std::setprecision(2) << millis[0] << " ms -> "
                                << millis[1] << " ms: "
                                << "engine = \"DEPTHWISE_3x3\"" << std::endl;
                      best_net.Clear();
                      best_net.CopyFrom(candidate_net);
                    }
                  }
                }
              }

              if (kernel[0] <= 8 && kernel[1] <= 8) {
                try_nnpack_convolution(
                    inputs,
                    init_net,
                    best_net,
                    op_index,
                    "FT8",
                    "PRECOMPUTE",
                    1);
              }

              if (kernel[0] <= 16 && kernel[1] <= 16) {
                try_nnpack_convolution(
                    inputs,
                    init_net,
                    best_net,
                    op_index,
                    "FT16",
                    "PRECOMPUTE",
                    1);
              }

              if (kernel[0] == 1 && kernel[1] == 1) {
                try_nnpack_convolution(
                    inputs, init_net, best_net, op_index, "DIRECT", "", 1);
              }
            } else if (stride[0] == 2 && stride[1] == 2) {
              /* Consider NNPACK with WINOGRAD convolution */
              if (kernel[0] == 3 && kernel[1] == 3) {
                try_nnpack_convolution(
                    inputs,
                    init_net,
                    best_net,
                    op_index,
                    "WINOGRAD",
                    "PRECOMPUTE",
                    1);
              }
            }

            try_nnpack_convolution(
                inputs,
                init_net,
                best_net,
                op_index,
                "IMPLICIT_GEMM",
                "PRECOMPUTE",
                1);
          }
        }
      }
    }
  }
  WriteProtoToBinaryFile(best_net, caffe2::FLAGS_optimized_net);
  return 0;
}

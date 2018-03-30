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

#include "caffe2/contrib/tensorrt/tensorrt_op_trt.h"
#include "caffe2/core/logging.h"

#include <numeric>
#include <unordered_map>

namespace caffe2 {

namespace {
// Note that input of trt tensor is in CHW format, while our tensor is NCHW
// \return -1 if there is dimension mismatch between C2 tensor and trt tensor.
// Otherwise, return the multiplicaton of CHW dimensions
int64_t CheckDims(
    const nvinfer1::Dims& nv_dims,
    const std::vector<TIndex>& c2_dims) {
  if (nv_dims.nbDims + 1 != c2_dims.size()) {
    return -1;
  }
  int64_t chw = 1;
  for (int i = 0; i < nv_dims.nbDims; ++i) {
    if (nv_dims.d[i] != c2_dims[i + 1]) {
      return -1;
    }
    chw *= nv_dims.d[i];
  }
  return chw;
}

} // namespace

// Upon construction, we build the inference enigne by deserializing from
// protobuf string. And since we know the input/output blobs, we can do the
// binding here too.
TensorRTOp::TensorRTOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CUDAContext>(operator_def, ws),
      logger_((nvinfer1::ILogger::Severity)(
          OperatorBase::GetSingleArgument<int>("log_verbosity", 2))),
      max_batch_size_(OperatorBase::GetSingleArgument<int>("max_batch_size", 1)) {
  {
    auto engine_string =
        OperatorBase::GetSingleArgument<std::string>("serialized_engine", "");
    CAFFE_ENFORCE(!engine_string.empty(), "Empty serialized TensorRT engine!");
    auto trt_runtime = InferObject(nvinfer1::createInferRuntime(logger_));
    // TODO(support trt plugin factory)
    trt_engine_ = InferObject(trt_runtime->deserializeCudaEngine(
        engine_string.data(), engine_string.size(), nullptr));
  }

  if(!trt_engine_) {
    CAFFE_THROW("Cannot deserialize TensorRT engine!");
  }

  std::unordered_map<std::string, int> inputs;
  std::unordered_map<std::string, int> outputs;
  for (int i = 0; i < operator_def.input_size(); ++i) {
    inputs.emplace(operator_def.input(i), i);
    VLOG(0) << "Adding Input: " << operator_def.input(i);
  }
  for (int i = 0; i < operator_def.output_size(); ++i) {
    outputs.emplace(operator_def.output(i), i);
    VLOG(0) << "Adding Output: " << operator_def.output(i);
  }

  // Set up the output size hints
  std::vector<int> output_size_hints_encoded(
      OperatorBase::GetRepeatedArgument<int>("output_size_hints"));
  std::vector<std::string> output_size_names(
      OperatorBase::GetRepeatedArgument<std::string>("output_size_names"));
  int idx = 0;
  for (const auto& oname : output_size_names) {
    const auto it = outputs.find(oname);
    if (it != outputs.end()) {
      std::vector<TIndex> dims;
      for (; idx < output_size_hints_encoded.size() && output_size_hints_encoded[idx] > 0; ++idx) {
        dims.push_back(output_size_hints_encoded[idx]);
      }
      output_size_hints_.emplace(it->second, std::move(dims));
    }
  }

  // match and bind the input/output
  int num_bindings = trt_engine_->getNbBindings();
  for (int b = 0; b < num_bindings; ++b) {
    const auto& name = trt_engine_->getBindingName(b);
    nv_dims_.push_back(trt_engine_->getBindingDimensions(b));
    if (trt_engine_->bindingIsInput(b)) {
      const auto it = inputs.find(name);
      CAFFE_ENFORCE(it != inputs.end(), MakeString("Cannot find trt input: ", name));
      binding_hints_.emplace_back(it->second, true);
    } else {
      const auto it = outputs.find(name);
      CAFFE_ENFORCE(it != outputs.end());
      binding_hints_.emplace_back(it->second, false);
    }
  }

  trt_executor_ = InferObject(trt_engine_->createExecutionContext());
}

void TensorRTOp::MaybeAdjustOutputShape(int output_idx, std::vector<TIndex>* dims) {
  const auto it = output_size_hints_.find(output_idx);
  if (it != output_size_hints_.end()) {
    const auto& dims_hint = it->second;
    auto total_trt = std::accumulate(dims->begin(), dims->end(), (TIndex)(1), std::multiplies<TIndex>());
    auto total_c2 = std::accumulate(dims_hint.begin(), dims_hint.end(), (TIndex)(1), std::multiplies<TIndex>());
    if (total_c2 != total_trt) {
      LOG(WARNING) << "The total size of TensorRT op output and hint don't match: " << total_trt << " vs " << total_c2;
      return;
    }

    bool identical_shape = true;
    if (dims->size() != dims_hint.size()) {
      identical_shape = false;
    } else {
      for (int i = 0; i < dims->size(); ++i) {
        if((*dims)[i] != dims_hint[i]) {
          identical_shape = false;
          break;
        }
      }
    }

    // We conform to the output shape hints. NB: We might need an explicit reshape op for this
    if (!identical_shape) {
      *dims = dims_hint;
    }
  }
}

bool TensorRTOp::RunOnDevice() {
  CAFFE_ENFORCE(trt_executor_);
  // Decide input batch size
  size_t N = 0;
  bool first = true;
  for (int i = 0; i < InputSize(); ++i) {
    const auto& input_tensor = Input(i);
    const auto& tensor_dims = input_tensor.dims();
    if (first) {
      N = tensor_dims.front();
      first = false;
    } else {
      CAFFE_ENFORCE_EQ(
          N, tensor_dims.front(), "Mismatched batch size in input tensors");
    }
  }

  // We need to do the binding at RunOnDevice time because we only know the
  // exact shapes of the tensors now. In addtion, since TensorRT engine has
  // max_batch_size, we need to call that multiple times if input batch size
  // exceeeds this limit.
  std::vector<void*> bindings;
  auto batch_size = max_batch_size_;
  for (size_t offset = 0; offset < N; offset += batch_size) {
    bindings.clear();
    batch_size =
        offset + max_batch_size_ > N ? N - offset : max_batch_size_;
    VLOG(2) << "Offset: " << offset << ", batch_size: " << batch_size << ", N: " << N;
    int b = 0;
    for (const auto& p : binding_hints_) {
      const auto& dims = nv_dims_[b++];
      if (p.second) {
        // input, check input dimensions
        const auto& input_tensor = Input(p.first);
        const float* input_data = input_tensor.data<float>();
        const auto& tensor_dims = input_tensor.dims();
        auto chw = CheckDims(dims, tensor_dims);
        CAFFE_ENFORCE_GE(chw, 0, "Mismatched dimensions between TRT input and C2 input");
        bindings.push_back((void*)(input_data + offset * chw));
      } else {
        // output, we need to allocate the output tensor at first batch run
        auto* output_tensor = Output(p.first);
        std::vector<TIndex> tensor_dims;
        tensor_dims.push_back(N);
        int64_t chw = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
          tensor_dims.push_back(dims.d[i]);
          chw *= dims.d[i];
        }

        if (offset == 0) {
          MaybeAdjustOutputShape(p.first, &tensor_dims);
          output_tensor->Resize(tensor_dims);
        }
        float* output_data = output_tensor->mutable_data<float>();
        bindings.push_back((void*)(output_data + offset * chw));
      }
    }

    CAFFE_ENFORCE(bindings.size() == InputSize() + OutputSize());
    if(!trt_executor_->execute(batch_size, &bindings[0])){
      return false;
    }
  }
  return true;
}

OPERATOR_SCHEMA(TensorRT)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
The TensorRT operator is a black-box operator serialized from prebuilt TensorRT
Engine string. It will take the input, do the computation by calling TensorRT
inference engine and generate the outputs.

This is a GPU only operator.
)DOC")
    .Arg(
        "log_verbosity",
        "(int default 0) verbosity of the TensorRt engine log."
        )
    .Arg(
        "serialized_engine",
        "(string default=\"\" blob for serialized TensorRT engine."
        "Note that serialized engine is not compatible across platform and "
        "different TensorRT version."
        )
    .Arg(
        "batch_size",
        "(int default 0) Batch size set by the TensorRT engine builder."
        "It must be no larger than the max_batch_size of the engine builder so "
        "it is better not to edit this manually.");

REGISTER_CUDA_OPERATOR(TensorRT, TensorRTOp);
} // namespace caffe2

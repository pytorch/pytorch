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

#include <functional>

#include "caffe2/core/init.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/fc_inference.h"
#include "caffe2/quantization/server/fb_fc_packed_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    FbFCPacked,
    FbFCPackedOperator<CPUContext, DefaultEngine, fbgemm::float16>);

using namespace std::placeholders;

vector<int64_t>
GetFbgemmTensorInfo(const void* c, size_t* capacity, DeviceOption* device) {
  const unique_ptr<fbgemm::PackedGemmMatrixFP16>* tc =
      static_cast<const unique_ptr<fbgemm::PackedGemmMatrixFP16>*>(c);
  device->set_device_type(PROTO_CPU);
  *capacity = (*tc)->numRows() * (*tc)->numCols() * 2;
  return {(*tc)->numCols(), (*tc)->numRows()};
}
bool Caffe2InitializeFbgemm(int*, char***) {
  RegisterTensorInfoFunction(
      TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>(),
      GetFbgemmTensorInfo);
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(
    InitFbgemmContext,
    &Caffe2InitializeFbgemm,
    "Register the tensor info function for the packed gemm matrix used in Fbgemm");

bool PackedGemmMatrixFP16ShapeFunctions::IsSameMetaType(TypeIdentifier id) {
  return id == TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>();
}

TypeIdentifier PackedGemmMatrixFP16ShapeFunctions::GetTypeMetaId() {
  return TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>();
}

TypeMeta PackedGemmMatrixFP16ShapeFunctions::GetExternalTensorType(
    const void* /* unused */) {
  return TypeMeta::Make<at::Half>();
}

vector<int64_t> PackedGemmMatrixFP16ShapeFunctions::GetExternalTensorInfo(
    const void* c,
    size_t* capacity,
    DeviceOption* device) {
  return GetFbgemmTensorInfo(c, capacity, device);
}

void PackedGemmMatrixFP16ShapeFunctions::SetupExternalTensorDescriptor(
    const Blob* blob,
    std::vector<std::vector<uint64_t>>* shapes,
    std::vector<std::vector<float>>* /* unused */,
    std::vector<std::vector<int32_t>>* /* unused */,
    ExternalTensorDescriptor* desc) {
  const auto* packed =
      blob->template Get<unique_ptr<fbgemm::PackedGemmMatrixFP16>>().get();

  // setup data and type
  desc->dataType = 10; // ONNXIFI_DATATYPE_FLOAT16
  desc->buffer = reinterpret_cast<uint64_t>(packed->pmat());

  // setup dim and shape
  std::vector<uint64_t> shape{static_cast<uint64_t>(packed->numCols()),
                              static_cast<uint64_t>(packed->numRows())};
  shapes->emplace_back(std::move(shape));
  desc->dimensions = 2;
  desc->shape = shapes->back().data();

  // no quantization params as this is not quantization
  desc->quantizationParams = 0;

  // not an offline tensor
  desc->isOffline = 0;
}

REGISTER_EXTERNAL_TENSOR_FUNCTIONS(
    (TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>()),
    PackedGemmMatrixFP16ShapeFunctions);

OPERATOR_SCHEMA(FbFCPacked)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        std::bind(CostInferenceForFC, _1, _2, false)))
    .SetDoc(R"DOC(Same as FC,
      but the weight is prepacked as a fbgemm::PackedGemmMatrixFP16)DOC");

} // namespace caffe2

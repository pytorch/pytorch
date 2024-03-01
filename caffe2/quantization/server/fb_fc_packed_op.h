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

#pragma once
#include <fbgemm/FbgemmFP16.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

/**
 * C2 wrapper for fp16 gemm
 *
 * Suppose your predict_net has an FC operator in fp32 as follows:
 * op {
 *   input: "x"
 *   input: "w"
 *   input: "b"
 *   output: "y"
 *   type: "FC"
 * }
 * ...
 * external_input: "w"
 *
 * To use FbFCPacked operator with fp16 fbgemm, in init_net
 * ... # an operator that generates w
 * op {
 *   input: "w"
 *   output: "w_packed"
 *   type: "FbGemmPack"
 * }
 * ...
 * external_output: "w_packed"
 *
 * in predict_net:
 * op {
 *   input: "x"
 *   input: "w_packed"
 *   input: "b"
 *   output: "y"
 *   type: "FbFCPacked"
 * }
 * ...
 * external_input: "w_packed"
 */
template <
    class Context,
    class Engine = DefaultEngine,
    typename T_W = fbgemm::float16>
class FbFCPackedOperator final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FbFCPackedOperator(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)) {}
  ~FbFCPackedOperator() override {}

  // template on X, B, and Y.
  template <typename T_X, typename T_B, typename T_Y>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& b = Input(2);

    CAFFE_ENFORCE(b.dim() == 1, b.dim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int M = X.size_to_dim(canonical_axis);
    const int N = b.numel();

    // Load the packed matrix
    auto* W =
        OperatorBase::Input<caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16>>(1)
            .get();
    const int K = W->numRows();
    if (!W->packed()) {
      if (!packed_w_) {
        std::vector<float> src_mat(W->matSize());
        for (int i = 0; i < W->matSize(); ++i) {
          src_mat[i] =
            fbgemm::cpu_half2float(W->pmat()[i]);
        }
        packed_w_ = std::make_unique<fbgemm::PackedGemmMatrixFP16>(
            fbgemm::matrix_op_t::Transpose,
            W->numRows(), W->numCols(),
            1.0,
            src_mat.data());
      }
      W = packed_w_.get();
    }

    auto dimErrorString = [&]() {
      return c10::str(
          "Dimension mismatch: ",
          "X: ",
          X.sizes(),
          ", W: ",
          std::vector<int>({K, W->numCols()}),
          ", b: ",
          b.sizes(),
          ", axis: ",
          axis_,
          ", M: ",
          M,
          ", N: ",
          N,
          ", K: ",
          K);
    };
    // Error checking
    CAFFE_ENFORCE(M == X.numel() / K, dimErrorString());
    CAFFE_ENFORCE(K == X.size_from_dim(canonical_axis), dimErrorString());
    CAFFE_ENFORCE(N == W->numCols(), dimErrorString());
    Y_shape_cache_ = X.sizes().vec();
    // This is an invariant of canonical_axis, so we can DCHECK.
    TORCH_DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
    Y_shape_cache_.resize(canonical_axis + 1);
    Y_shape_cache_[canonical_axis] = N;
    auto* Y = Output(0, Y_shape_cache_, at::dtype<T_Y>());

    if (X.numel() == 0) {
      // skip the rest of the computation if X is empty
      Y->template mutable_data<T_Y>();
      return true;
    }

    // Call the fp16 gemm interface
    fbgemm::cblas_gemm_compute(
        fbgemm::matrix_op_t::NoTranspose,
        M,
        X.template data<T_X>(),
        *W,
        0.f,
        Y->template mutable_data<T_Y>());

    // Add bias term, accumulation is still in fp32.
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T_B, Context>(
          M,
          convert::To<float, T_B>(1),
          bias_multiplier_.template mutable_data<T_B>(),
          &context_);
    }
    math::Gemm<T_B, Context, Engine>(
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        1,
        1,
        bias_multiplier_.template data<T_B>(),
        b.template data<T_B>(),
        1,
        Y->template mutable_data<T_Y>(),
        &context_,
        math_type);

    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<
        float, // X
        float, // B
        float>(); // Y
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_;
  Tensor bias_multiplier_{Context::GetDeviceType()};
  caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16> packed_w_{nullptr};
};

class PackedGemmMatrixFP16ShapeFunctions : public ExternalTensorFunctionsBase {
 public:
  explicit PackedGemmMatrixFP16ShapeFunctions()
      : ExternalTensorFunctionsBase() {}
  ~PackedGemmMatrixFP16ShapeFunctions() override {}
  bool isQuantized() const override {
    return false;
  }
  bool IsSameMetaType(TypeIdentifier id) override;
  void SetupExternalTensorDescriptor(
      const Blob* blob,
      std::vector<std::vector<uint64_t>>* shapes,
      std::vector<std::vector<float>>* all_scales,
      std::vector<std::vector<int32_t>>* all_offsets,
      ExternalTensorDescriptor* desc) override;
  void LoadInfoOfBlob(
      const Blob* /* unused */,
      std::vector<float>* /* unused */,
      std::vector<float>* /* unused */,
      uint32_t* /* unused */) override {}
  TypeIdentifier GetTypeMetaId() override;
  TypeMeta GetExternalTensorType(const void* c) override;
  vector<int64_t> GetExternalTensorInfo(
      const void* c,
      size_t* capacity,
      DeviceOption* device) override;
};

} // namespace caffe2

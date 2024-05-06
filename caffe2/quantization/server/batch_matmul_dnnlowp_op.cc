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

#include "batch_matmul_dnnlowp_op.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// #define DNNLOWP_MEASURE_TIME_BREAKDOWN
#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

namespace caffe2 {

using namespace std;
using namespace dnnlowp;

template <typename T>
BatchMatMulDNNLowPOp<T>::BatchMatMulDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      trans_a_(this->template GetSingleArgument<int>("trans_a", 0)),
      trans_b_(this->template GetSingleArgument<int>("trans_b", 0)),
      broadcast_(this->template GetSingleArgument<int>("broadcast", 0)),
      is_B_constant_(
          this->template GetSingleArgument<bool>("constant_B", false)) {}

template <typename T>
bool BatchMatMulDNNLowPOp<T>::RunOnDevice() {
  this->ParseDNNLowPOperatorArguments_();

  const auto& A = InputTensorCPU_(0);
  const auto& B = InputTensorCPU_(1);
  auto* Y = OutputTensorCPU_(0);

  auto ndims_A = A.ndim();
  auto dims_A = A.sizes().vec();
  auto ndims_B = B.ndim();
  auto dims_B = B.sizes().vec();

  auto noBroadcastErrorMsg = [](size_t dim1, size_t dim2) {
    std::stringstream ss;
    ss << "Inputs with dimensions A = ";
    ss << dim1;
    ss << " and B = ";
    ss << dim2;
    ss << " is not supported with broadcast=0. Did you forget to set the "
          "broadcast flag?";
    return ss.str();
  };

  // These should all be false if we're not broadcasting.
  bool dimMismatch = ndims_A != ndims_B;
  bool dimsLessThan1D = ndims_A < 2;
  CAFFE_ENFORCE(
      broadcast_ || (!dimMismatch && !dimsLessThan1D),
      noBroadcastErrorMsg(ndims_A, ndims_B));

  auto dimMismatchErrorString = [](size_t dimnum1,
                                   size_t dim1,
                                   size_t dimnum2,
                                   size_t dim2,
                                   bool trans_a,
                                   bool trans_b) {
    std::stringstream ss;
    ss << "Expected dimension ";
    ss << dimnum1;
    ss << " of tensor A with value ";
    ss << dim1;
    ss << " to match dimension ";
    ss << dimnum2;
    ss << " of tensor B with value ";
    ss << dim2;
    ss << ". trans_a = ";
    ss << trans_a;
    ss << " trans_b = ";
    ss << trans_b;
    return ss.str();
  };

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int num_sub_batches, num_outer_batches;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t M, N, K;
  size_t A_stride = 1; // How far to increment A pointer each itr
  size_t B_stride = 1; // How far to increment B pointer each itr
  size_t Y_stride = 1; // How far to increment Y pointer each itr
  if (ndims_A == 1 && ndims_B == 1) {
    // vector-vector
    CAFFE_ENFORCE_EQ(
        dims_A[0],
        dims_B[0],
        "Vector-vector product requires each of the vectors to "
        "be the same size.");
    Y->Resize(1);
    num_sub_batches = 1;
    num_outer_batches = 1;
    M = 1;
    N = 1;
    K = dims_A[0];
  } else {
    bool A_broadcasted = false, B_broadcasted = false;
    if (ndims_A == 1) {
      dims_A.insert(dims_A.begin(), 1);
      ndims_A = 2;
      A_broadcasted = true;
    }
    if (ndims_B == 1) {
      dims_B.push_back(1);
      ndims_B = 2;
      B_broadcasted = true;
    }
    // matrix-matrix with batches
    // [B1..., M, K] * [B2..., K, N] -> [B..., M, N]
    // In the event that A or B are one-dimensional, the trailing or leading
    // 1 is not added to the output tensor's size.

    // First step: partition the tensors into inner and outer blocks.
    // Ignoring the last two dimensions of A and B, ensure that one of the
    // tensors' dimensions is a suffix of the other. For example,
    // [4, x, x] is a suffix of [2, 3, 4, x, x]. In this example, the
    // dimensions of size 2 and 3 will be broadcasted, so we partition into
    // 2*3=6 individual instances of batched GEMM with A and B \in [4, x, x].
    size_t num_inner_dims = std::min(ndims_A, ndims_B);
    for (size_t i = 2; i < num_inner_dims; ++i) {
      auto first_r_itr = dims_A.rbegin();
      auto second_r_itr = dims_B.rbegin();
      CAFFE_ENFORCE_EQ(
          *(first_r_itr + i),
          *(second_r_itr + i),
          dimMismatchErrorString(
              ndims_A - i - 1,
              *(first_r_itr + i),
              ndims_B - i - 1,
              *(second_r_itr + i),
              trans_a_,
              trans_b_));
    }
    size_t num_outer_dims = std::max(ndims_A, ndims_B) - num_inner_dims;

    // Standard M, N, and K parameters respecting GEMM API and transpose
    // flags
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t K_dim;
    if (trans_a_) {
      M = dims_A[ndims_A - 1];
      K = dims_A[ndims_A - 2];
      K_dim = ndims_A - 2;
    } else {
      M = dims_A[ndims_A - 2];
      K = dims_A[ndims_A - 1];
      K_dim = ndims_A - 1;
    }
    if (trans_b_) {
      N = dims_B[ndims_B - 2];
      CAFFE_ENFORCE_EQ(
          K,
          dims_B[ndims_B - 1],
          dimMismatchErrorString(
              K_dim, K, ndims_B - 1, dims_B[ndims_B - 1], trans_a_, trans_b_));
    } else {
      N = dims_B[ndims_B - 1];
      CAFFE_ENFORCE_EQ(
          K,
          dims_B[ndims_B - 2],
          dimMismatchErrorString(
              K_dim, K, ndims_B - 2, dims_B[ndims_B - 2], trans_a_, trans_b_));
    }

    // Calculate output tensor shapes [B..., (M), (N)]
    // Batch dimensions will be broadcasted out to those of the longer tensor
    // A or B. Either M or N are optional if A or B, respectively are 1-D.
    std::vector<int64_t> new_dims;
    if (ndims_A >= ndims_B) {
      new_dims.assign(dims_A.begin(), dims_A.end() - 2);
    } else {
      new_dims.assign(dims_B.begin(), dims_B.end() - 2);
    }
    if (!A_broadcasted) {
      new_dims.push_back(M);
    } else {
      new_dims.push_back(1);
    }
    if (!B_broadcasted) {
      new_dims.push_back(N);
    } else {
      new_dims.push_back(1);
    }

    // Calculate strides. Continuing our example above,
    //   [4, M, K] * [2, 3, 4, K, N] = [2, 3, 4, M, N]
    // We calculate this as follows:
    //   1) Treat the outer batch dimensions as flattened, i.e. view the B
    //      tensor here as [6, 4, K, N] and Y as [6, 4, M, N]. The same rea-
    //      soning is analogous for the case where # dims A >= # dims B.
    //   2) Perform this operation:
    //        for i in range(6):
    //          Y[i, :, :, :] = BatchMatMul(A, B[i, :, :, :])
    A_stride = 1; // How far to increment A pointer each itr
    B_stride = 1; // How far to increment B pointer each itr
    Y_stride = 1; // How far to increment Y pointer each itr
    // How many "inner batches" we have. That is, the product of sizes for
    // the slices excluding M, K, and N, for their respective matrices.
    num_sub_batches = 1;
    if (ndims_A >= ndims_B) {
      auto first_r_itr = dims_A.rbegin();
      auto output_r_itr = new_dims.rbegin();
      for (size_t i = 0; i < num_inner_dims; ++i) {
        A_stride *= *(first_r_itr + i);
        Y_stride *= *(output_r_itr + i);
        if (i >= 2) {
          num_sub_batches *= *(first_r_itr + i);
        }
      }
      B_stride = 0;
    } else {
      A_stride = 0;
      auto second_r_itr = dims_B.rbegin();
      auto output_r_itr = new_dims.rbegin();
      for (size_t i = 0; i < num_inner_dims; ++i) {
        B_stride *= *(second_r_itr + i);
        Y_stride *= *(output_r_itr + i);
        if (i >= 2) {
          num_sub_batches *= *(second_r_itr + i);
        }
      }
    }

    num_outer_batches = 1;
    for (size_t i = 0; i < num_outer_dims; ++i) {
      num_outer_batches *= new_dims[i];
    }

    // Mutually exclusive since otherwise we would've taken the vector-vector
    // path above
    if (A_broadcasted) {
      new_dims.erase(new_dims.end() - 2);
    } else if (B_broadcasted) {
      new_dims.erase(new_dims.end() - 1);
    }

    // Allocate output tensor
    Y->Resize(new_dims);

    // Optimize case num_sub_batches == 1 where we can combine batched gemms
    // into a single gemm
    if (num_sub_batches == 1 && num_outer_batches > 1) {
      if (ndims_A > ndims_B && !trans_a_) {
        M *= num_outer_batches;
        num_outer_batches = 1;
      }
    }
  }

  // Zero batch dimension indicates no elements
  if (num_sub_batches == 0 || num_outer_batches == 0) {
    if (dequantize_output_) {
      Y->template mutable_data<float>();
    } else {
      Y->template mutable_data<T>();
    }
    return true;
  }

  // Choose quantization for X
  in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
  int num_batches_B = B.numel() / (K * N);
  if (!first_invocation_ && !Bq_packed_.empty() &&
      num_batches_B * N != column_offsets_.size()) {
    LOG(INFO) << "Operator with output " << this->debug_def().output(0)
              << " does not have constant B";
    is_B_constant_ = false;
    Bq_packed_.clear();
  }
  bool fast_path =
      std::is_same<T, uint8_t>::value && GetCpuId().avx2() && is_B_constant_;

  if (fast_path) {
    // Quantize B
    if (Bq_packed_.empty()) {
      int signed_min = -(1 << (qfactory_->GetWeightPrecision() - 1));
      vector<int8_t> B_quantized_temp(K * N);
      column_offsets_.resize(num_batches_B * N);
      for (int i = 0; i < num_batches_B; ++i) {
        if (this->template InputIsType<int8::Int8TensorCPU>(1)) {
          // NOLINTNEXTLINE(modernize-use-emplace)
          B_qparams_.push_back(TensorQuantizationParams());
          B_qparams_[i].scale =
              this->template Input<int8::Int8TensorCPU>(1).scale;
          B_qparams_[i].zero_point =
              this->template Input<int8::Int8TensorCPU>(1).zero_point +
              signed_min;

          const T* B_data = B.template data<T>() + i * B_quantized_temp.size();
          // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
          for (auto j = 0; j < B_quantized_temp.size(); ++j) {
            B_quantized_temp[j] = B_data[j] + signed_min;
          }
        } else {
          B_qparams_.emplace_back(qfactory_->ChooseQuantizationParams(
              B.template data<float>() + i * B_quantized_temp.size(),
              B_quantized_temp.size(),
              true /* weight */));

          // B_qparams_[i] is computed for unsigned type.
          // Adjust for the fact that B will actually use signed.
          B_qparams_[i].zero_point += signed_min;

          fbgemm::Quantize<int8_t>(
              B.template data<float>() + i * B_quantized_temp.size(),
              B_quantized_temp.data(),
              B_quantized_temp.size(),
              B_qparams_[i]);
        }

        Bq_packed_.emplace_back(new fbgemm::PackBMatrix<int8_t>(
            trans_b_ ? fbgemm::matrix_op_t::Transpose
                     : fbgemm::matrix_op_t::NoTranspose,
            K,
            N,
            B_quantized_temp.data(),
            trans_b_ ? K : N,
            nullptr /*pmat*/,
            1)); /*groups*/

        // Pre-compute column_offset
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int j = 0; j < N; ++j) {
          int32_t sum = 0;
          if (trans_b_) {
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            for (int k = 0; k < K; ++k) {
              sum += B_quantized_temp[j * K + k];
            }
          } else {
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            for (int k = 0; k < K; ++k) {
              sum += B_quantized_temp[k * N + j];
            }
          }
          column_offsets_[i * N + j] = sum - B_qparams_[i].zero_point * K;
        }
      } // for each input in the batch
    } // Bq_packed_.empty()

    if (!dequantize_output_) {
      GetOutputQuantizationParams_();

      for (int i = 0; i < num_batches_B; ++i) {
        float real_multiplier =
            in_qparams_[0].scale * B_qparams_[i].scale / out_qparams_.scale;
        requantization_params_.emplace_back(
            qfactory_->ChooseRequantizationMultiplier(
                real_multiplier, out_qparams_));
      }
    } else {
      if (measure_quantization_error_) {
        // to measure quantization error, run ref impl.
        Fp32Op_()->DequantizeInput();
        Fp32Op_()->Get()->RunOnDevice();
      }
    }
  } else {
    // slow path
    if (first_invocation_) {
      string reason;
      if (!is_same<T, uint8_t>::value) {
        reason = "fbgemm only supports 8-bit integers";
      } else if (!GetCpuId().avx2()) {
        reason = "fbgemm only supports AVX2";
      } else if (!is_B_constant_) {
        reason = "B is not constant";
      } else {
        assert(false);
      }
      LOG(WARNING) << "BatchMatMul with output " << this->debug_def().output(0)
                   << " falls back to slow path because " << reason;
    }
    B_qparams_.resize(1);
    requantization_params_.resize(1);

    B_qparams_[0] =
        GetInputTensorQuantizationParamsOf(this, 1, qfactory_.get());

    GetOutputQuantizationParams_();

    float real_multiplier =
        in_qparams_[0].scale * B_qparams_[0].scale / out_qparams_.scale;
    requantization_params_[0] = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);
  }

  first_invocation_ = false;

  vector<T> A_temp, B_temp;
  if (!Bq_packed_.empty()) {
    // fast path
    using namespace fbgemm;

    const T* A_quantized = nullptr;
    if (A.template IsType<T>() || !dequantize_output_) {
      // Only when input and output are float, we don't need input to be
      // quantized.
      A_quantized = QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], A_temp);
    }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    chrono::time_point<chrono::system_clock> t_begin, t_end;
    t_begin = chrono::system_clock::now();
#endif

    if (!dequantize_output_) {
      auto Y_data = Y->template mutable_data<T>();

      auto row_offset_len_per_thread =
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize();
      row_offsets_.resize(
          row_offset_len_per_thread * dnnlowp_get_max_threads());
      auto A_pack_buf_len_per_thread =
          PackAWithRowOffset<uint8_t>::packedBufferSize();
      A_pack_buf_.resize(A_pack_buf_len_per_thread * dnnlowp_get_max_threads());
      Y_int32_.resize(Y->numel());

#ifdef _OPENMP
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(2)
#endif
#endif
      for (int p = 0; p < num_outer_batches; ++p) {
        for (int i = 0; i < num_sub_batches; ++i) {
          int tid = dnnlowp_get_thread_num();

          PackAWithRowOffset<uint8_t> packA(
              trans_a_ ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
              M,
              K,
              reinterpret_cast<const uint8_t*>(A_quantized) + p * A_stride +
                  i * M * K,
              trans_a_ ? M : K,
              A_pack_buf_.data() +
                  tid * A_pack_buf_len_per_thread, // buffer for packed matrix
              1, // group
              row_offsets_.data() + tid * row_offset_len_per_thread);

          int B_batch_idx = ndims_A >= ndims_B ? i : p * num_sub_batches + i;
          DoNothing<> doNothingObj{};
          ReQuantizeOutput<false /* FUSE_RELU */> outputProcObj(
              doNothingObj,
              &requantization_params_[B_batch_idx].real_multiplier,
              out_qparams_.zero_point,
              in_qparams_[0].zero_point,
              &B_qparams_[B_batch_idx].zero_point,
              packA.getRowOffsetBuffer(),
              column_offsets_.data() + B_batch_idx * N,
              nullptr, // bias
              N); // ncols per quant group

          fbgemmPacked(
              packA,
              *Bq_packed_[B_batch_idx],
              reinterpret_cast<uint8_t*>(Y_data) + p * Y_stride + i * M * N,
              Y_int32_.data() + p * Y_stride + i * M * N,
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        } // for each input in batch
      }

      PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
    } else {
      // dequantize_output
      float* Y_data = Y->template mutable_data<float>();

      if (!A.template IsType<T>()) {
        // Both input and output are float
        int row_offset_len_per_thread =
            PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize();
        row_offsets_.resize(
            row_offset_len_per_thread * dnnlowp_get_max_threads());
        int A_pack_len_per_thread =
            PackAWithQuantRowOffset<uint8_t>::packedBufferSize();
        A_pack_buf_.resize(A_pack_len_per_thread * dnnlowp_get_max_threads());

#ifdef _OPENMP
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(2)
#endif
#endif
        for (int p = 0; p < num_outer_batches; ++p) {
          for (int i = 0; i < num_sub_batches; ++i) {
            int tid = dnnlowp_get_thread_num();

            PackAWithQuantRowOffset<uint8_t> packA(
                trans_a_ ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
                M,
                K,
                A.template data<float>() + p * A_stride + i * M * K,
                trans_a_ ? M : K,
                A_pack_buf_.data() +
                    tid * A_pack_len_per_thread, // buffer for packed matrix
                in_qparams_[0].scale,
                in_qparams_[0].zero_point,
                1, // groups
                row_offsets_.data() + tid * row_offset_len_per_thread);

            int B_batch_idx = ndims_A >= ndims_B ? i : p * num_sub_batches + i;
            DoNothing<float, float> doNothingObj{};
            ReQuantizeForFloat<false /* FUSE_RELU*/> outputProcObj(
                doNothingObj,
                in_qparams_[0].scale,
                &B_qparams_[B_batch_idx].scale,
                in_qparams_[0].zero_point,
                &B_qparams_[B_batch_idx].zero_point,
                packA.getRowOffsetBuffer(),
                column_offsets_.data() + B_batch_idx * N,
                nullptr, // bias
                N); // ncols per quant group

            fbgemmPacked(
                packA,
                *Bq_packed_[B_batch_idx],
                Y_data + p * Y_stride + i * M * N,
                reinterpret_cast<int32_t*>(Y_data) + p * Y_stride + i * M * N,
                N,
                outputProcObj,
                0, // thread_id
                1); // num_threads
          } // for each input in batch
        }
      } else {
        // Input quantized and output float
        auto row_offset_len_per_thread =
            PackAWithRowOffset<uint8_t>::rowOffsetBufferSize();
        row_offsets_.resize(
            row_offset_len_per_thread * dnnlowp_get_max_threads());
        auto A_pack_buf_len_per_thread =
            PackAWithRowOffset<uint8_t>::packedBufferSize();
        A_pack_buf_.resize(
            A_pack_buf_len_per_thread * dnnlowp_get_max_threads());

#ifdef _OPENMP
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(2)
#endif
#endif
        for (int p = 0; p < num_outer_batches; ++p) {
          for (int i = 0; i < num_sub_batches; ++i) {
            int tid = dnnlowp_get_thread_num();

            PackAWithRowOffset<uint8_t> packA(
                trans_a_ ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
                M,
                K,
                reinterpret_cast<const uint8_t*>(A_quantized) + p * A_stride +
                    i * M * K,
                trans_a_ ? M : K,
                A_pack_buf_.data() +
                    tid * A_pack_buf_len_per_thread, // buffer for packed matrix
                1, // group
                row_offsets_.data() + tid * row_offset_len_per_thread);

            int B_batch_idx = ndims_A >= ndims_B ? i : p * num_sub_batches + i;
            DoNothing<float, float> doNothingObj{};
            ReQuantizeForFloat<false /* FUSE_RELU*/> outputProcObj(
                doNothingObj,
                in_qparams_[0].scale,
                &B_qparams_[B_batch_idx].scale,
                in_qparams_[0].zero_point,
                &B_qparams_[B_batch_idx].zero_point,
                packA.getRowOffsetBuffer(),
                column_offsets_.data() + B_batch_idx * N,
                nullptr, // bias
                N); // ncols per quant group

            fbgemmPacked(
                packA,
                *Bq_packed_[B_batch_idx],
                Y_data + p * Y_stride + i * M * N,
                reinterpret_cast<int32_t*>(Y_data) + p * Y_stride + i * M * N,
                N,
                outputProcObj,
                0, // thread_id
                1); // num_threads
          } // for each input in batch
        }
      }
    } // dequantize_output

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    double gops =
        2. * num_outer_batches * num_sub_batches * M * N * K / dt / 1e9;
    LOG(INFO) << "batches " << num_outer_batches * num_sub_batches << " m " << M
              << " n " << N << " k " << K << " " << gops << " gops";
#endif

    MeasureQuantizationError_();
  } else {
    // slow path
    // Quantize inputs
    const T* A_quantized =
        QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], A_temp);
    const T* B_quantized =
        QuantizeInputIfNeeded<T>(this, 1, B_qparams_[0], B_temp);

    T* Y_quantized = GetQuantizedOutputData_();
    Y_int32_.resize(Y->numel());
#ifdef _OPENMP
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(2)
#endif
#endif
    for (int p = 0; p < num_outer_batches; ++p) {
      for (int i = 0; i < num_sub_batches; ++i) {
        // Y_q = (scale_A * scale_B) / scale_Y * Y_int32
        // Y_int32 = (A_q - zero_point_A * 1_A) * (B_q - zero_point_B * 1_B),
        //           where 1_A is a matrix with all 1s and same size as A
        // Y_int32 = A_q * B_q
        //           - zero_point_A * 1_A * B - zero_point_B * A * 1_B
        //           + zero_point_A * zero_point_B * 1_A * 1_B
        // zero_point_A * 1_A * B : a matrix with (i, j) is the sum of jth
        //                          column of B. This is computed by
        //                          column_offsets in the code.
        // zero_point_B * A * 1_B : a matrix with (i, j) is the sum of ith row
        //                          of A. This is computed by row_offset in the
        //                          code.
        // zero_point_A * zero_point_B * 1_A * 1_B : a matrix with all elements
        //                          are zero_point_A * zero_point_B *
        //                          num_of_cols_of_A. This is computed by
        //                          const_offset in the code.
        const T* A_quantized_i = A_quantized + p * A_stride + i * M * K;
        const T* B_quantized_i = B_quantized + p * B_stride + i * K * N;

        int32_t const_offset =
            in_qparams_[0].zero_point * B_qparams_[0].zero_point * K;
        vector<int32_t> column_offsets(N);
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int n = 0; n < N; ++n) {
          int32_t sum = 0;
          if (trans_b_) {
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            for (int k = 0; k < K; ++k) {
              sum += B_quantized_i[k + n * K];
            }
          } else {
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            for (int k = 0; k < K; ++k) {
              sum += B_quantized_i[k * N + n];
            }
          }
          column_offsets[n] = sum * in_qparams_[0].zero_point;
        }

        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int m = 0; m < M; ++m) {
          int32_t row_offset = 0;
          if (trans_a_) {
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            for (int k = 0; k < K; ++k) {
              row_offset += A_quantized_i[m + k * M];
            }
          } else {
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            for (int k = 0; k < K; ++k) {
              row_offset += A_quantized_i[m * K + k];
            }
          }
          row_offset *= B_qparams_[0].zero_point;

          // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
          for (int n = 0; n < N; ++n) {
            int32_t sum = 0;
            if (!trans_a_ && !trans_b_) {
              // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
              for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A_quantized_i[m * K + k]) *
                    static_cast<int32_t>(B_quantized_i[k * N + n]);
              }
            } else if (!trans_a_ && trans_b_) {
              // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
              for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A_quantized_i[m * K + k]) *
                    static_cast<int32_t>(B_quantized_i[k + n * K]);
              }
            } else if (trans_a_ && !trans_b_) {
              // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
              for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A_quantized_i[m + k * M]) *
                    static_cast<int32_t>(B_quantized_i[k * N + n]);
              }
            } else if (trans_a_ && trans_b_) {
              // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
              for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A_quantized_i[m + k * M]) *
                    static_cast<int32_t>(B_quantized_i[k + n * K]);
              }
            }

            Y_int32_[p * Y_stride + i * M * N + m * N + n] =
                sum - row_offset - column_offsets[n] + const_offset;
          } // for each output col
        } // for each output row

        // Requantization
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int j = 0; j < M * N; ++j) {
          Y_quantized[p * Y_stride + i * M * N + j] = fbgemm::Requantize<T>(
              Y_int32_[p * Y_stride + i * M * N + j],
              requantization_params_[0]);
        }
      } // for each batch
    }

    RunOnDeviceEpilogue_();
  }

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    BatchMatMul,
    DNNLOWP,
    BatchMatMulDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    BatchMatMul,
    DNNLOWP_16,
    BatchMatMulDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8BatchMatMul,
    DNNLOWP,
    BatchMatMulDNNLowPOp<uint8_t>);

} // namespace caffe2

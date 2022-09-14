#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input TensorViews and the function will
// create the correct intermediate nodes and return the output TensorViews.
//

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct ForwardNormResult {
  TensorView* output = nullptr;
  TensorView* mean = nullptr;
  TensorView* invstd = nullptr;
};

struct BackwardNormResult {
  TensorView* grad_input = nullptr;
  TensorView* grad_weight = nullptr;
  TensorView* grad_bias = nullptr;
};

struct ForwardRMSNormResult {
  TensorView* output = nullptr;
  TensorView* invstd = nullptr;
};

struct BackwardRMSNormResult {
  TensorView* grad_input = nullptr;
  TensorView* grad_weight = nullptr;
};

struct VarMeanResult {
  TensorView* var = nullptr;
  TensorView* mean = nullptr;
};

TORCH_CUDA_CU_API TensorView* mean(
    TensorView* x,
    const std::vector<int>& dims,
    bool keepdim);

TORCH_CUDA_CU_API TensorView* variance(
    TensorView* x,
    const std::vector<int>& dims,
    bool unbiased,
    bool keepdim);

TORCH_CUDA_CU_API TensorView* variance(
    TensorView* x,
    const std::vector<int>& dims,
    int64_t correction,
    bool keepdim);

TORCH_CUDA_CU_API VarMeanResult variance_mean(
    TensorView* x,
    const std::vector<int>& dims,
    int64_t correction,
    bool keepdim);

TORCH_CUDA_CU_API TensorView* standard_deviation(
    TensorView* x,
    const std::vector<int>& dims,
    bool unbiased,
    bool keepdim);

TORCH_CUDA_CU_API TensorView* softmax(TensorView* x, int dim);

TORCH_CUDA_CU_API TensorView* softmax_backward(
    TensorView* dy,
    TensorView* y,
    const int dim);

TORCH_CUDA_CU_API TensorView* log_softmax(TensorView* x, int dim);

TORCH_CUDA_CU_API TensorView* log_softmax_backward(
    TensorView* dy,
    TensorView* y,
    const int dim);

TORCH_CUDA_CU_API ForwardNormResult layer_norm(
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    TensorView* bias,
    Val* eps);

TORCH_CUDA_CU_API ForwardNormResult layer_norm(
    TensorView* x,
    const size_t kNormShapeNumDims,
    TensorView* weight,
    TensorView* bias,
    Val* eps);

TORCH_CUDA_CU_API ForwardRMSNormResult rms_norm(
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    Val* eps);

TORCH_CUDA_CU_API ForwardRMSNormResult rms_norm(
    TensorView* x,
    const size_t kNormShapeNumDims,
    TensorView* weight,
    Val* eps);

TORCH_CUDA_CU_API BackwardNormResult layer_norm_backward(
    TensorView* dy,
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* mean,
    TensorView* rstd,
    TensorView* weight,
    TensorView* bias,
    const std::vector<bool>& output_mask);

TORCH_CUDA_CU_API BackwardRMSNormResult rms_norm_backward(
    TensorView* dy,
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* rstd,
    TensorView* weight,
    const std::vector<bool>& output_mask);

TORCH_CUDA_CU_API ForwardNormResult batch_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kTraining,
    Val* momentum,
    Val* eps,
    bool channels_last = false);

TORCH_CUDA_CU_API BackwardNormResult batch_norm_backward(
    TensorView* x,
    TensorView* dy,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask,
    bool channels_last = false);

TORCH_CUDA_CU_API ForwardNormResult instance_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kUseInputStats, // kTraining?
    Val* momentum,
    Val* eps,
    bool channels_last = false);

TORCH_CUDA_CU_API BackwardNormResult instance_norm_backward(
    TensorView* x,
    TensorView* dy,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask,
    bool channels_last = false);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

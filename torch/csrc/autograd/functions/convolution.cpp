#include "convolution.h"

#include <sstream>

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/utils/auto_gpu.h"

#include "ATen/Tensor.h"

#ifdef WITH_CUDNN
#include "torch/csrc/cudnn/Conv.h"
#include "torch/csrc/cudnn/Handles.h"
#include "torch/csrc/cudnn/Types.h"
extern THCState* state;
using namespace torch::cudnn;
#endif

#ifdef WITH_NNPACK
#include "torch/csrc/nnpack/NNPACK.h"
#endif

using torch::cudnn::Convolution;
using at::Tensor;
using tensor_pair = std::pair<at::Tensor, at::Tensor>;

namespace torch { namespace autograd {

// Forward function definition and utility functions

static at::Tensor compute_output(
  at::Tensor& input, at::Tensor& weight, at::Tensor& bias, at::Tensor& columns, at::Tensor& ones,
  const std::vector<int64_t>& kernel_size, const ConvForward& params);

static at::Tensor compute_grad_input(
  at::Tensor& input, at::Tensor& grad_output, at::Tensor& weight, at::Tensor& columns, at::Tensor& ones,
  const std::vector<int64_t>& kernel_size, const ConvParams& params);

static tensor_pair compute_grad_params(
  at::Tensor& input, at::Tensor& grad_output, at::Tensor& weight, at::Tensor& bias, at::Tensor& columns, at::Tensor& ones,
  const std::vector<int64_t>& kernel_size, const ConvBackward& params);

auto ConvParams::is_strided() const -> bool {
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

auto ConvParams::is_dilated() const -> bool {
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

auto ConvParams::is_output_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_output_padding_big() const -> bool {
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |= (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

auto ConvParams::is_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}


auto ConvParams::view1d_as_2d() -> void {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

auto ConvParams::use_cudnn(const at::Tensor& input) const -> bool {
#ifdef WITH_CUDNN
  if (!input.type().isCuda() || !cudnn_enabled) {
    return false;
  }
  if (is_dilated()) {
    cudaDeviceProp* prop = THCState_getCurrentDeviceProperties(state);
    // NOTE: extra parenthesis around numbers disable clang warnings about dead code
    return ((CUDNN_VERSION >= (6021)) || (CUDNN_VERSION >= (6000) && prop->major >= 5)) && !is_output_padding_big();
  }
  return !is_output_padding_big();
#endif
  return false;
}

auto ConvParams::use_nnpack(const at::Tensor& input) const -> bool {
#ifdef WITH_NNPACK
  return input.type().ID() == at::TypeID::CPUFloat && // only on CPU Float Tensors
         !is_strided() && // doesn't support strides
         !is_dilated() && // or dilation
         !transposed &&   // or transposed tensors
         input.ndimension() == 4 && // must be in NCHW format
         input.size(0) >= 16; // ensure large enough batch size to ensure perf, tuneable
#endif
  return false;
}

std::string ConvForward::name() { return "ConvForward"; }

auto ConvForward::output_size(at::Tensor& input, at::Tensor& weight) const -> std::vector<int64_t> {
  auto in_size = input.sizes();
  auto weight_size = weight.sizes();
  auto dim = input.ndimension();

  std::vector<int64_t> output_size(dim);
  output_size[0] = in_size[0];
  output_size[1] = transposed ? weight_size[1] * groups : weight_size[0];
  for (int d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    if (transposed) {
      output_size[d] = (in_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                       kernel + output_padding[d - 2];
    } else {
      output_size[d] = (in_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
    }
  }
  return output_size;
}

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  if (tensor.ndimension() != 3) throw std::runtime_error("expected 3D tensor");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  if (tensor.ndimension() != 4) throw std::runtime_error("expected 4D tensor");
  return tensor.squeeze(2);
}

static void check_input_shape_forward(const at::Tensor& input,
				      const at::Tensor& weight,
				      int64_t groups, bool transposed) {
  if (!transposed) {
    if (input.size(1) != (weight.size(1) * groups)) {
      std::stringstream ss;
      ss << "Given groups=" << groups << ", weight" << weight.sizes()
	 << ", so expected input" << input.sizes() << "  to have "
	 << (weight.size(1) * groups) << " channels, but got " << input.size(1)
	 << " channels instead";
      throw std::runtime_error(ss.str());
    }
  } else { // transposed
    if (input.size(1) != weight.size(0)) {
      std::stringstream ss;
      ss << "Given transposed=" << transposed << ", weight" << weight.sizes()
	 << ", so expected input" << input.sizes() << "  to have "
	 << weight.size(0) << " channels, but got " << input.size(1)
	 << " channels instead";
      throw std::runtime_error(ss.str());
    }
  }
}

static at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

static Variable subvariable(const Variable& var, int dim, int groups, int g) {
  int64_t n = var.sizes()[dim] / groups;
  auto result = apply_fn<Narrow>(dim, n * g, n)(var);
  return result;
}

static at::Tensor cat(const tensor_list& tensors, int dim) {
  int num_inputs = tensors.size();
  if (num_inputs == 0) {
    return at::Tensor();
  }

  auto output = tensors[0].type().tensor();
  at::cat_out(output, tensors, dim);
  return output;
}


// ConvForward implementation

auto ConvForward::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("ConvNd", inputs, 3, 2);
  if (is_padding_neg()) throw std::runtime_error("negative padding is not supported");
  if (is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");

  AutoGPU guard(inputs[0]);

  auto input = inputs[0].data().contiguous();
  auto weight = inputs[1].data();
  auto bias = inputs[2].opt_data();

  check_input_shape_forward(input, weight, groups, transposed);

  int k = input.ndimension();
  if (k == 3) {
    view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  auto weight_size = weight.sizes();
  std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());

  auto output = input.type().tensor();
  tensor_list columns(groups);
  tensor_list ones(groups);
  std::unique_ptr<Convolution> convolution;

  if (use_cudnn(input)) {
#ifdef WITH_CUDNN
    if (input.type().ID() != weight.type().ID()){
      std::stringstream ss;
      ss << "Input type (" << input.toString() << ") and weight type (" << weight.toString() << ") should be the same";
      throw std::runtime_error(ss.str());
    }
    if (bias.defined() && input.type().ID() != bias.type().ID()){
      std::stringstream ss;
      ss << "Input type (" << input.toString() << ") and bias type (" << bias.toString() << ") should be the same";
      throw std::runtime_error(ss.str());
    }

    output = input.type().tensor();
    output.resize_(output_size(input, weight));
    if (transposed) {
      convolution.reset(cudnn_convolution_transpose_full_forward(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(input),
          (THVoidTensor*)input.unsafeGetTH(false), (THVoidTensor*)weight.unsafeGetTH(false),
          bias.defined() ? (THVoidTensor*)bias.unsafeGetTH(false) : nullptr, (THVoidTensor*)output.unsafeGetTH(false),
          padding, stride, dilation, groups, benchmark));
    } else {
      convolution.reset(cudnn_convolution_full_forward(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(input),
          (THVoidTensor*)input.unsafeGetTH(false), (THVoidTensor*)weight.unsafeGetTH(false),
          bias.defined() ? (THVoidTensor*)bias.unsafeGetTH(false) : nullptr, (THVoidTensor*)output.unsafeGetTH(false),
          padding, stride, dilation, groups, benchmark));
    }
#endif
  } else {
    for (int g = 0; g < groups; ++g) {
      columns[g] = input.type().tensor();
      ones[g] = input.type().tensor();
    }
    if (groups == 1) {
      output = compute_output(
          input, weight, bias,
          columns[0], ones[0], kernel_size, *this);
    } else {
      tensor_list outputs(groups);
      for (int g = 0; g < groups; ++g) {
        auto input_g = subtensor(input, 1, groups, g);
        auto weight_g = subtensor(weight, 0, groups, g);
        auto bias_g = subtensor(bias, 0, groups, g);
        outputs[g] = compute_output(
            input_g, weight_g, bias_g,
            columns[g], ones[g], kernel_size, *this);
      }
      output = cat(outputs, 1);
    }
  }

  if (k == 3) {
    output = view3d(output);
  }

  auto outputs = as_tensor_list(std::move(output));
  return wrap_outputs(inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<ConvBackward>(
        f, *this,
        inputs[0], inputs[1], inputs[2],
        std::move(columns), std::move(ones), std::move(convolution));
  });
};


// ConvBackward implementation

auto ConvBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("ConvNdBackward", grad_outputs, 1);
  if (is_padding_neg()) throw std::runtime_error("negative padding is not supported");
  if (is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");

  auto input_var = input_.unpack();
  auto weight_var = weight_.unpack();
  auto bias_var = bias_.unpack();

  auto input = input_var.data();
  auto weight = weight_var.data();

  AutoGPU guard(input);

  auto bias = bias_var.defined() ? bias_var.data() : Tensor();

  input = input.contiguous();
  auto grad_output = grad_outputs[0].data().contiguous();

  int k = input.ndimension();
  if (k == 3) {
    input = view4d(input);
    weight = view4d(weight);
    grad_output = view4d(grad_output);
  }

  auto weight_size = weight.sizes();
  std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());

  bool use_cudnn = this->use_cudnn(input);

  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  if (should_compute_output(0)) {
    if (use_cudnn) {
#ifdef WITH_CUDNN
      grad_input = input.type().tensor();
      grad_input.resize_as_(input);
      if (transposed) {
        // ConvTranspose uses the same kernels as regular convolution
        // but swaps forward and backward calls
        cudnn_convolution_forward(
            state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(input),
            (THVoidTensor*)grad_output.unsafeGetTH(false), (THVoidTensor*)weight.unsafeGetTH(false), (THVoidTensor*)grad_input.unsafeGetTH(false),
            convolution.get(), benchmark);
      } else {
        cudnn_convolution_backward_data(
            state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(input),
            (THVoidTensor*)grad_output.unsafeGetTH(false), (THVoidTensor*)grad_input.unsafeGetTH(false), (THVoidTensor*)weight.unsafeGetTH(false),
            convolution.get(), benchmark);
      }
#endif
    } else if (groups == 1) {
      grad_input = compute_grad_input(
          input, grad_output, weight,
          columns[0], ones[0], kernel_size, *this);
    } else {
      tensor_list grad_inputs(groups);
      for (int g = 0; g < groups; ++g) {
        auto input_g = subtensor(input, 1, groups, g);
        auto grad_output_g = subtensor(grad_output, 1, groups, g);
        auto weight_g = subtensor(weight, 0, groups, g);
        grad_inputs[g] = compute_grad_input(
            input_g, grad_output_g, weight_g,
            columns[g], ones[g], kernel_size, *this);
      }
      grad_input = cat(grad_inputs, 1);
    }
  }

  if (should_compute_output(1) || should_compute_output(2)) {
    if (use_cudnn) {
#ifdef WITH_CUDNN
      grad_weight = weight.type().tensor();
      grad_weight.resize_as_(weight);
      cudnn_convolution_backward_filter(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(input),
          (THVoidTensor*)grad_output.unsafeGetTH(false), (THVoidTensor*)input.unsafeGetTH(false), (THVoidTensor*)grad_weight.unsafeGetTH(false),
          convolution.get(), benchmark);

      if (bias.defined() && should_compute_output(2)) {
        grad_bias = bias.type().tensor();
        grad_bias.resize_as_(bias);
        cudnn_convolution_backward_bias(
            state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(input),
            (THVoidTensor*)grad_output.unsafeGetTH(false), (THVoidTensor*)grad_bias.unsafeGetTH(false),
            convolution.get());
      }
#endif
    } else if (groups == 1) {
      std::tie(grad_weight, grad_bias) = compute_grad_params(
          input, grad_output, weight, bias,
          columns[0], ones[0], kernel_size, *this);
    } else {
      tensor_list grad_weights(groups);
      tensor_list grad_biases(groups);
      for (int g = 0; g < groups; ++g) {
        auto input_g = subtensor(input, 1, groups, g);
        auto grad_output_g = subtensor(grad_output, 1, groups, g);
        auto weight_g = subtensor(weight, 0, groups, g);
        auto bias_g = subtensor(bias, 0, groups, g);
        std::tie(grad_weights[g], grad_biases[g]) = compute_grad_params(
            input_g, grad_output_g, weight_g, bias_g,
            columns[g], ones[g], kernel_size, *this);
      }
      grad_weight = cat(grad_weights, 0);
      if (bias.defined() && should_compute_output(2)) {
        grad_bias = cat(grad_biases, 0);
      }
    }
  }

  if (k == 3) {
    if (grad_input.defined()) {
      grad_input = view3d(grad_input);
    }
    if (grad_weight.defined()) {
      grad_weight = view3d(grad_weight);
    }
  }

  // Add saved variables used out of the pure autograd to inputs
  variable_list all_inputs(grad_outputs);
  all_inputs.push_back(input_var);
  all_inputs.push_back(weight_var);

  auto outputs =  as_tensor_list(std::move(grad_input),
                                 std::move(grad_weight),
                                 std::move(grad_bias));
  return wrap_outputs(all_inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<ConvBackwardBackward>(
      f, *this,
      input_var, weight_var,
      bias_var, grad_outputs[0]);
  });
};

auto ConvBackward::releaseVariables() -> void {
  input_.data.reset();
  weight_.data.reset();
  bias_.data.reset();
}


// ConvBackwardBackward implementation

auto ConvBackwardBackward::apply(const variable_list& grad_grad_inputs) -> variable_list {
  check_input_variables("ConvNdBackwardBackward", grad_grad_inputs, 3, 0);
  if (transposed) throw std::runtime_error("ConvBackwardBackward does not support transposed convolution");

  auto ggI = grad_grad_inputs[0];
  auto ggW = grad_grad_inputs[1];
  auto ggb = grad_grad_inputs[2];

  auto gO = grad_output_.unpack();
  auto weight = weight_.unpack();
  auto input = input_.unpack();

  AutoGPU guard(input.data());

  // Compute ggO = conv(w, ggI) + conv(ggW, i) + ggb
  Variable ggO;
  if (ggI.defined()) {
    if (weight.type().isCuda()) {
      weight = apply_fn<Contiguous>()(weight);
    }
    ggO = apply_fn<ConvForward>(*this)(ggI, weight, Variable());
  }

  if (ggW.defined()) {
    if (ggW.type().isCuda()) {
      ggW = apply_fn<Contiguous>()(ggW);
    }
    auto ggW_term = apply_fn<ConvForward>(*this)(input_.unpack(), ggW, Variable());
    if (ggO.defined()) {
      ggO = apply_fn<Add>()(ggO, ggW_term);
    } else {
      ggO = ggW_term;
    }
  }

  if (ggb.defined()) {
    // View as (1, ggb.size(0), 1, 1...)

    // Expand
    std::vector<int64_t> new_size(gO.ndimension(), 1);
    new_size[1] = ggb.sizes()[0];
    auto ggb_contiguous = apply_fn<Contiguous>()(ggb);
    auto ggb_view = apply_fn<View>(new_size)(ggb_contiguous);

    // Expand
    auto ggb_expanded = apply_fn<Expand>(gO.sizes())(ggb_view);

    if (ggO.defined()) {
      ggO = apply_fn<Add>()(ggO, ggb_expanded);
    } else {
      ggO = ggb_expanded;
    }
  }

  // Compute gW = conv(ggI, g0)
  Variable gW;
  if (ggI.defined()) {
    // Modified params with correct padding
    ConvParams gw_conv_params(*this);

    // Disable groups as they are handled separately
    auto groups = gw_conv_params.groups;
    gw_conv_params.groups = 1;
    std::swap(gw_conv_params.dilation, gw_conv_params.stride);

    // Transpose gO and ggI to accumulate over batch
    auto gOt = apply_fn<Transpose>(0, 1)(gO);
    auto ggIt = apply_fn<Transpose>(0, 1)(ggI);

    Variable gWt;
    // Compute conv
    if (groups == 1) {
      if (gOt.type().isCuda()) {
        gOt = apply_fn<Contiguous>()(gOt);
      }

      // Compute conv
      gWt = apply_fn<ConvForward>(gw_conv_params)(ggIt, gOt, Variable());
    } else {
      variable_list gWt_list(groups);
      for (int g = 0; g < groups; ++g) {
        auto ggIt_g = subvariable(ggIt, 0, groups, g);
        auto gOt_g = subvariable(gOt, 0, groups, g);
        if (gOt_g.type().isCuda()) {
          gOt_g = apply_fn<Contiguous>()(gOt_g);
        }

        gWt_list[g] = apply_fn<ConvForward>(gw_conv_params)(ggIt_g, gOt_g, Variable());
      }

      gWt = apply_fn<Cat>(1)(gWt_list);
    }

    // Transpose gW to match chan_in and chan_out
    gW = apply_fn<Transpose>(0, 1)(gWt);

    // narrow gW to only relevant portion
    // we do it this way instead of narrowing the input itself because
    // the ConvForward kernels don't support asymmetric padding.
    auto gW_size = gW.sizes();
    auto w_size = weight.sizes();
    for (size_t i = 2; i < gW_size.size(); ++i) {
      if (gW_size[i] > w_size[i]) {
          gW = apply_fn<Narrow>(i, 0, w_size[i])(gW);
      }
    }
  }

  // Compute gI = convT(gO, ggW)
  Variable gI;
  if (ggW.defined()) {
    // select conv transpose
    ConvParams gi_conv_params(*this);
    gi_conv_params.transposed = true;

    // swap stride and dilation
    std::swap(gi_conv_params.dilation, gi_conv_params.stride);

    // calculate output_padding
    auto weight_size = weight.sizes();
    std::vector<long> kernel_size(weight_size.begin() + 2, weight_size.end());
    auto input_size = input.sizes();
    std::vector<long> input_shape(input_size.begin() + 2, input_size.end());
    auto grad_output_size = gO.sizes();
    std::vector<long> grad_output_shape(grad_output_size.begin() + 2, grad_output_size.end());

    if (kernel_size.size() == 1) {
      auto expected_input_shape = (kernel_size[0] - 1) * gi_conv_params.stride[1]
          - 2 * gi_conv_params.padding[1]
          + (gi_conv_params.dilation[1] * (grad_output_shape[0] - 1) + 1);
      if (expected_input_shape != input_shape[0]) {
          gi_conv_params.output_padding[1] = input_shape[0] - expected_input_shape;
      }
    } else {
      for(size_t i = 0; i < kernel_size.size(); ++i) {
        // Check if whole input has been used or not
        auto expected_input_shape = (kernel_size[i] - 1) * gi_conv_params.stride[i]
          - 2 * gi_conv_params.padding[i]
          + (gi_conv_params.dilation[i] * (grad_output_shape[i] - 1) + 1);
        if (expected_input_shape != input_shape[i]) {
          gi_conv_params.output_padding[i] = input_shape[i] - expected_input_shape;
        }
      }
    }

    // Disable groups as they are handled separately
    auto groups = gi_conv_params.groups;
    gi_conv_params.groups = 1;

    auto ggWt = apply_fn<Transpose>(0, 1)(ggW);
    auto gOt = apply_fn<Transpose>(0, 1)(gO);

    Variable gIt;
    if (groups == 1) {
      if (gOt.type().isCuda()) {
        gOt = apply_fn<Contiguous>()(gOt);
      }

      gIt = apply_fn<ConvForward>(gi_conv_params)(ggWt, gOt, Variable());
    } else {
      variable_list gIt_list(groups);
      for (int g = 0; g < groups; ++g) {
        auto ggWt_g = subvariable(ggWt, 1, groups, g);
        auto gOt_g = subvariable(gOt, 0, groups, g);
        if (gOt_g.type().isCuda()) {
          gOt_g = apply_fn<Contiguous>()(gOt_g);
        }

        gIt_list[g] = apply_fn<ConvForward>(gi_conv_params)(ggWt_g, gOt_g, Variable());
      }

      gIt = apply_fn<Cat>(0)(gIt_list);
    }

    gI = apply_fn<Transpose>(0, 1)(gIt);
  }

  return {ggO, gI, gW};
}

auto ConvBackwardBackward::releaseVariables() -> void {
  input_.data.reset();
  weight_.data.reset();
  bias_.data.reset();
  grad_output_.data.reset();
}

// Forward and backward functions for Tensor

static at::Tensor compute_output(
    at::Tensor& input, at::Tensor& weight, at::Tensor& bias,
    at::Tensor& columns, at::Tensor& ones,
    const std::vector<int64_t>& kernel_size,
    const ConvForward& params) {

  auto output = input.type().tensor();
  auto dim = input.ndimension();
  auto dilated = params.is_dilated();


  if (params.transposed) {
    if (dim == 4) {
      at::SpatialFullDilatedConvolution_updateOutput(
          input, output, weight, bias, columns, ones,
          kernel_size[1], kernel_size[0],
          params.stride[1], params.stride[0],
          params.padding[1], params.padding[0],
          dilated ? params.dilation[1] : 1,
          dilated ? params.dilation[0] : 1,
          params.output_padding[1], params.output_padding[0]); goto done;
    } else if (dim == 5) {
      at::VolumetricFullDilatedConvolution_updateOutput(
          input, output, weight, bias, columns, ones,
          params.stride[0], params.stride[2], params.stride[1],
          params.padding[0], params.padding[2], params.padding[1],
          dilated ? params.dilation[0] : 1,
          dilated ? params.dilation[2] : 1,
          dilated ? params.dilation[1] : 1,
          params.output_padding[0], params.output_padding[2], params.output_padding[1]); goto done;
      }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        at::SpatialDilatedConvolution_updateOutput(
          input, output, weight, bias, columns, ones,
          kernel_size[1], kernel_size[0],
          params.stride[1], params.stride[0],
          params.padding[1], params.padding[0],
          params.dilation[1], params.dilation[0]); goto done;
      } else {  /* dim == 4, non-dilated */
        if (params.use_nnpack(input)) {
#ifdef WITH_NNPACK
          // THNN functions handle resizing the output Tensor themselves,
          // but NNPACK expects the Tensors to be in the appropriate shape
          // already, so we resize here
          output.resize_(params.output_size(input, weight));
          nnpack::SpatialConvolution_updateOutput(
              input, output, weight, bias,
              kernel_size[1], kernel_size[0],
              params.padding[1], params.padding[0]); goto done;
#endif
        } else {
        /* CPU implementation has specialized MM kernels
           for non-dilated case here */
        at::SpatialConvolutionMM_updateOutput(
            input, output, weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
        }
      }
    } else if (dim == 5 && (input.type().isCuda() || dilated)) {
      at::VolumetricDilatedConvolution_updateOutput(
          input, output, weight, bias, columns, ones,
          kernel_size[0], kernel_size[2], kernel_size[1],
          params.stride[0], params.stride[2], params.stride[1],
          params.padding[0], params.padding[2], params.padding[1],
          dilated ? params.dilation[0] : 1,
          dilated ? params.dilation[2] : 1,
          dilated ? params.dilation[1] : 1); goto done;
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
      /* CPU implementation has specialized MM kernels
         for non-dilated case here */
      at::VolumetricConvolutionMM_updateOutput(
          input, output, weight, bias, columns,
          kernel_size[0], kernel_size[2], kernel_size[1],
          params.stride[0], params.stride[2], params.stride[1],
          params.padding[0], params.padding[2], params.padding[1]); goto done;
    }
  }

  throw std::runtime_error("unsupported ConvNd parameters");

done:
  return output;
}

static at::Tensor compute_grad_input(
    at::Tensor& input, at::Tensor& grad_output, at::Tensor& weight, at::Tensor& columns, at::Tensor& ones,
    const std::vector<int64_t>& kernel_size, const ConvParams& params) {

  auto grad_input = input.type().tensor();
  grad_input.resize_as_(input);
  auto dim = input.ndimension();
  auto dilated = params.is_dilated();

  if (params.transposed) {
    if (dim == 4) {
      at::SpatialFullDilatedConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            dilated ? params.dilation[1] : 1,
            dilated ? params.dilation[0] : 1,
            params.output_padding[1], params.output_padding[0]); goto done;
    } else if (dim == 5) {
      at::VolumetricFullDilatedConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            dilated ? params.dilation[0] : 1,
            dilated ? params.dilation[2] : 1,
            dilated ? params.dilation[1] : 1,
            params.output_padding[0], params.output_padding[2], params.output_padding[1]); goto done;
    }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        at::SpatialDilatedConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0]); goto done;
      } else {
        if (params.use_nnpack(input)) {
#ifdef WITH_NNPACK
          nnpack::SpatialConvolution_updateGradInput(
              input, grad_output, grad_input, weight,
              kernel_size[1], kernel_size[0],
              params.padding[1], params.padding[0]); goto done;
#endif
        } else {
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */
          at::SpatialConvolutionMM_updateGradInput(
              input, grad_output, grad_input, weight, columns, ones,
              kernel_size[1], kernel_size[0],
              params.stride[1], params.stride[0],
              params.padding[1], params.padding[0]); goto done;
        }
      }
    } else if (dim == 5 && (input.type().isCuda() || dilated)) {
        at::VolumetricDilatedConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            dilated ? params.dilation[0] : 1,
            dilated ? params.dilation[2] : 1,
            dilated ? params.dilation[1] : 1); goto done;
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU implementation has specialized MM kernels
           for non-dilated case here */
        at::VolumetricConvolutionMM_updateGradInput(
            input, grad_output, grad_input, weight, columns, ones,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
    }
  }

  throw std::runtime_error("unsupported ConvNdBackward parameters");

done:
  return grad_input;
}

static tensor_pair compute_grad_params(
    at::Tensor& input, at::Tensor& grad_output, at::Tensor& weight, at::Tensor& bias,
    at::Tensor& columns, at::Tensor& ones,
    const std::vector<int64_t>& kernel_size, const ConvBackward& params) {

  auto grad_weight = weight.type().tensor();
  grad_weight.resize_as_(weight).zero_();

  at::Tensor grad_bias;
  if (bias.defined() && params.should_compute_output(2)) {
    grad_bias = bias.type().tensor();
    grad_bias.resize_as_(bias).zero_();
  }

  auto dim = input.ndimension();
  auto dilated = params.is_dilated();

 if (params.transposed) {
    if (dim == 4) {
      at::SpatialFullDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            dilated ? params.dilation[1] : 1,
            dilated ? params.dilation[0] : 1,
            params.output_padding[1], params.output_padding[0], 1.0); goto done;
    } else if (dim == 5) {
        at::VolumetricFullDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            dilated ? params.dilation[0] : 1,
            dilated ? params.dilation[2] : 1,
            dilated ? params.dilation[1] : 1,
            params.output_padding[0], params.output_padding[2], params.output_padding[1], 1.0); goto done;
    }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        at::SpatialDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0], 1.0); goto done;
      } else {
        if (params.use_nnpack(input)) {
#ifdef WITH_NNPACK
          // NNPACK does not have a bias gradient calculation, so we split
          // into two calls here if necessary
          nnpack::SpatialConvolution_accGradWeight(
              input, grad_output, grad_weight,
              kernel_size[1], kernel_size[0],
              params.padding[1], params.padding[0]);

          if (bias.defined() && params.should_compute_output(2)) {
            // grad_output is in N, C, H, W, we re-shape and make contiguous
            at::Tensor ones = grad_output.type().ones(input.size(0) * grad_output.size(2) * grad_output.size(3));
            at::Tensor reshaped = grad_output.transpose(1, 3).contiguous().view({-1, ones.numel()});
            grad_bias.addmv_(1.0, 1.0, reshaped, ones);
          }
          goto done;
#endif
        } else {
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */
          at::SpatialConvolutionMM_accGradParameters(
              input, grad_output, grad_weight, grad_bias, columns, ones,
              kernel_size[1], kernel_size[0],
              params.stride[1], params.stride[0],
              params.padding[1], params.padding[0], 1.0); goto done;
        }
      }
    } else if (dim == 5 && (input.type().isCuda() || dilated)) {
        at::VolumetricDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            dilated ? params.dilation[0] : 1,
            dilated ? params.dilation[2] : 1,
            dilated ? params.dilation[1] : 1, 1.0); goto done;
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU implementation has specialized MM kernels
           for non-dilated case here */
        at::VolumetricConvolutionMM_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1], 1.0); goto done;
    }
  }

  throw std::runtime_error("unsupported ConvNdBackward parameters");

done:
  return std::make_pair<>(std::move(grad_weight), std::move(grad_bias));
}

}} // namespace torch::autograd

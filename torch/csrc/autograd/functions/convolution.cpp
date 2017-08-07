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

using torch::cudnn::Convolution;
using tensor_pair = std::pair<at::Tensor, at::Tensor>;

namespace torch { namespace autograd {

// Forward function definition and utility functions

static at::Tensor compute_output(
  at::Tensor& input, at::Tensor& weight, at::Tensor& bias, at::Tensor& columns, at::Tensor& ones,
  const std::vector<int64_t>& kernel_size, const ConvParams& params);

static at::Tensor compute_grad_input(
  at::Tensor& input, at::Tensor& grad_output, at::Tensor& weight, at::Tensor& columns, at::Tensor& ones,
  const std::vector<int64_t>& kernel_size, const ConvParams& params);

static tensor_pair compute_grad_params(
  at::Tensor& input, at::Tensor& grad_output, at::Tensor& weight, at::Tensor& bias, at::Tensor& columns, at::Tensor& ones,
  const std::vector<int64_t>& kernel_size, const ConvBackward& params);

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
    return ((CUDNN_VERSION >= (6021)) || (CUDNN_VERSION >= (6000) && prop->major >= 5)) && !transposed;
  }
  return true;
#endif
  return false;
}

auto ConvForward::output_size(at::Tensor& input, at::Tensor& weight) -> std::vector<int64_t> {
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

static at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

static std::shared_ptr<Variable> subvariable(std::shared_ptr<Variable> var, int dim, int groups, int g) {
  int64_t n = var->data.sizes()[dim] / groups;
  auto result = std::make_shared<Narrow>(dim, n * g, n)->apply({var})[0];
  return result;
}

static at::Tensor cat(const tensor_list& tensors, int dim) {
  int num_inputs = tensors.size();
  if (num_inputs == 0) {
    return at::Tensor();
  }

  auto output = tensors[0].type().tensor();
  at::cat_out(tensors, dim, output);
  return output;
}


// ConvForward implementation

auto ConvForward::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("ConvNd", inputs, 3, 2);
  if (is_padding_neg()) throw std::runtime_error("negative padding is not supported");
  if (is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");
  AutoGPU guard(inputs[0]->data);
  auto input = inputs[0]->data.contiguous();
  auto weight = inputs[1]->data;
  auto bias = inputs[2] ? inputs[2]->data : at::Tensor();

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
        inputs[0]->save(this), inputs[1]->save(this), Variable::save_opt(inputs[2].get(), this),
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

  auto input = input_var->data;
  auto weight = weight_var->data;
  auto bias = bias_var ? bias_var->data : at::Tensor();

  AutoGPU guard(input);

  input = input.contiguous();
  auto grad_output = grad_outputs[0]->data.contiguous();

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
      input_var->save(this), weight_var->save(this),
      Variable::save_opt(bias_var.get(), this), grad_outputs[0]->save(this));
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

  // Compute ggO = conv(w, ggI) + conv(ggW, i) + ggb
  std::shared_ptr<Variable> ggO = nullptr;
  if (ggI) {
    if (weight->data.type().isCuda()) {
      weight = Contiguous().apply({weight})[0];
    }
    ggO = ConvForward(*this).apply({ggI, weight, nullptr})[0];
  }

  if (ggW) {
    if (ggW->data.type().isCuda()) {
      ggW = Contiguous().apply({ggW})[0];
    }
    auto ggW_term = ConvForward(*this).apply({input_.unpack(), ggW, nullptr})[0];
    if (ggO) {
      ggO = Add().apply({ggO, ggW_term})[0];
    } else {
      ggO = ggW_term;
    }
  }

  if (ggb) {
    // View as (1, ggb.size(0), 1, 1...)

    // Expand
    std::vector<int64_t> new_size(gO->data.ndimension(), 1);
    new_size[1] = ggb->data.sizes()[0];
    auto ggb_contiguous = Contiguous().apply({ggb})[0];
    auto ggb_view = View(new_size).apply({ggb_contiguous})[0];

    // Expand
    auto ggb_expanded = Expand(gO->data.sizes()).apply({ggb_view})[0];

    if (ggO) {
      ggO = Add().apply({ggO, ggb_expanded})[0];
    } else {
      ggO = ggb_expanded;
    }
  }

  // Compute gW = conv(gO, ggI)
  std::shared_ptr<Variable> gW = nullptr;
  if (ggI) {
    // Modified params with correct padding
    ConvParams gw_conv_params(*this);
    // Disable groups as they are handled separately
    auto groups = gw_conv_params.groups;
    gw_conv_params.groups = 1;
    auto weight_size = weight->data.sizes();
    std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());
    auto input_size = ggI->data.sizes();
    std::vector<int64_t> input_shape(input_size.begin() + 2, input_size.end());
    for(size_t i=0; i<gw_conv_params.padding.size(); ++i) {
      // Check if whole input has been used or not
      auto numerator = 2 * gw_conv_params.padding[i] -
            gw_conv_params.dilation[i] * (kernel_size[i] - 1) - 1;
      auto remainder = (input_shape[i] + numerator) % gw_conv_params.stride[i];
      if (remainder != 0) {
        auto used_input_size = input_shape[i] - remainder;
        ggI = Narrow(i+2, 0, used_input_size).apply({ggI})[0];
      }
    }
    std::swap(gw_conv_params.dilation, gw_conv_params.stride);

    // Transpose gO and ggI to accumulate over batch
    auto gOt = Transpose(0, 1).apply({gO})[0];
    auto ggIt = Transpose(0, 1).apply({ggI})[0];

    std::shared_ptr<Variable> gWt = nullptr;
    // Compute conv
    if (groups == 1) {
      if (gOt->data.type().isCuda()) {
        gOt = Contiguous().apply({gOt})[0];
      }

      // Compute conv
      gWt = ConvForward(gw_conv_params).apply({ggIt, gOt, nullptr})[0];
    } else {
      variable_list gWt_list(groups);
      for (int g = 0; g < groups; ++g) {
        auto ggIt_g = subvariable(ggIt, 0, groups, g);
        auto gOt_g = subvariable(gOt, 0, groups, g);
        if (gOt_g->data.type().isCuda()) {
          gOt_g = Contiguous().apply({gOt_g})[0];
        }

        gWt_list[g] = ConvForward(gw_conv_params).apply({ggIt_g, gOt_g, nullptr})[0];
      }

      gWt = Cat(1).apply(gWt_list)[0];
    }

    // Transpose gW to match chan_in and chan_out
    gW = Transpose(0, 1).apply({gWt})[0];
  }

  // Compute gI = convT(gO, ggW)
  std::shared_ptr<Variable> gI = nullptr;
  if (ggW) {
    // select conv tranpose and swap stride and dilation
    ConvParams gi_conv_params(*this);
    gi_conv_params.transposed = true;
    // Disable groups as they are handled separately
    auto groups = gi_conv_params.groups;
    gi_conv_params.groups = 1;
    for(size_t i=0; i<gi_conv_params.padding.size(); ++i) {
      if (gi_conv_params.stride[i] != 1) {
        // TODO: Remove this when transpose dilated is fixed
        throw std::runtime_error("Second argument of ConvNdBackwardBackward is not zero."
        "This is not supported at the moment.");
      }
    }
    std::swap(gi_conv_params.dilation, gi_conv_params.stride);

    auto ggWt = Transpose(0, 1).apply({ggW})[0];
    auto gOt = Transpose(0, 1).apply({gO})[0];

    std::shared_ptr<Variable> gIt = nullptr;
    if (groups == 1) {
      if (gOt->data.type().isCuda()) {
        gOt = Contiguous().apply({gOt})[0];
      }

      gIt = ConvForward(gi_conv_params).apply({ggWt, gOt, nullptr})[0];
    } else {
      variable_list gIt_list(groups);
      for (int g = 0; g < groups; ++g) {
        auto ggWt_g = subvariable(ggWt, 1, groups, g);
        auto gOt_g = subvariable(gOt, 0, groups, g);
        if (gOt_g->data.type().isCuda()) {
          gOt_g = Contiguous().apply({gOt_g})[0];
        }

        gIt_list[g] = ConvForward(gi_conv_params).apply({ggWt_g, gOt_g, nullptr})[0];
      }

      gIt = Cat(0).apply(gIt_list)[0];
    }

    gI = Transpose(0, 1).apply({gIt})[0];
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
    const ConvParams& params) {

  auto output = input.type().tensor();
  auto dim = input.ndimension();
  auto dilated = params.is_dilated();

  if (dilated) {
    if (params.transposed) {
      /* dilated && transposed */
      /* NOT IMPLEMENTED */
    } else /* !transposed */ {
      /* dilated && !transposed */
      if (dim == 4) {
        at::SpatialDilatedConvolution_updateOutput(
            input, output, weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0]); goto done;
      } else if (dim == 5) {
        at::VolumetricDilatedConvolution_updateOutput(
            input, output, weight, bias, columns, ones,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.dilation[0], params.dilation[2], params.dilation[1]); goto done;
      }
    }
  } else /* !dilated */ {
    if (params.transposed) {
      /* !dilated && transposed */
      if (dim == 4) {
        at::SpatialFullConvolution_updateOutput(
            input, output, weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.output_padding[1], params.output_padding[0]); goto done;
      } else if (dim == 5) {
        at::VolumetricFullConvolution_updateOutput(
            input, output, weight, bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.output_padding[0], params.output_padding[2], params.output_padding[1]); goto done;
      }
    } else /* !transposed */ {
      /* !dilated && !transposed */
      if (dim == 4) {
        at::SpatialConvolutionMM_updateOutput(
            input, output, weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
      } else if (dim == 5 && input.type().isCuda()) {
        at::VolumetricConvolution_updateOutput(
            input, output, weight, bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
      } else if (dim == 5) {
        at::VolumetricConvolutionMM_updateOutput(
            input, output, weight, bias, columns,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
      }
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

  if (dilated) {
    if (params.transposed) {
      /* dilated && transposed */
      /* NOT IMPLEMENTED */
    } else /* !transposed */ {
      /* dilated && !transposed */
      if (dim == 4) {
        at::SpatialDilatedConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0]); goto done;
      } else if (dim == 5) {
        at::VolumetricDilatedConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.dilation[0], params.dilation[2], params.dilation[1]); goto done;
      }
    }
  } else /* !dilated */ {
    if (params.transposed) {
      /* !dilated && transposed */
      if (dim == 4) {
        at::SpatialFullConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.output_padding[1], params.output_padding[0]); goto done;
      } else if (dim == 5) {
        at::VolumetricFullConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.output_padding[0], params.output_padding[2], params.output_padding[1]); goto done;
      }
    } else /* !transposed */ {
      /* !dilated && !transposed */
      if (dim == 4) {
        at::SpatialConvolutionMM_updateGradInput(
            input, grad_output, grad_input, weight, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
      } else if (dim == 5 && input.type().isCuda()) {
        at::VolumetricConvolution_updateGradInput(
            input, grad_output, grad_input, weight, columns,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
      } else if (dim == 5) {
        at::VolumetricConvolutionMM_updateGradInput(
            input, grad_output, grad_input, weight, columns, ones,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
      }
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

  if (dilated) {
    if (params.transposed) {
      /* dilated && transposed */
      /* NOT IMPLEMENTED */
    } else /* !transposed */ {
      /* dilated && !transposed */
      if (dim == 4) {
        at::SpatialDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0], 1.0); goto done;
      } else if (dim == 5) {
        at::VolumetricDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.dilation[0], params.dilation[2], params.dilation[1], 1.0); goto done;
      }
    }
  } else /* !dilated */ {
    if (params.transposed) {
      /* !dilated && transposed */
      if (dim == 4) {
        at::SpatialFullConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.output_padding[1], params.output_padding[0], 1.0); goto done;
      } else if (dim == 5) {
        at::VolumetricFullConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.output_padding[0], params.output_padding[2], params.output_padding[1], 1.0); goto done;
      }
    } else /* !transposed */ {
      /* !dilated && !transposed */
      if (dim == 4) {
        at::SpatialConvolutionMM_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0], 1.0); goto done;
      } else if (dim == 5 && input.type().isCuda()) {
        at::VolumetricConvolution_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1], 1.0); goto done;
      } else if (dim == 5) {
        at::VolumetricConvolutionMM_accGradParameters(
            input, grad_output, grad_weight, grad_bias, columns,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1], 1.0); goto done;
      }
    }
  }
  throw std::runtime_error("unsupported ConvNdBackward parameters");

done:
  return std::make_pair<>(std::move(grad_weight), std::move(grad_bias));
}

}} // namespace torch::autograd

#include "convolution.h"

#include <sstream>

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/nn/THNN_generic.h"
#include "torch/csrc/utils/auto_gpu.h"

#include "THPP/Type.hpp"

#ifdef WITH_CUDNN
#include "torch/csrc/cudnn/Conv.h"
#include "torch/csrc/cudnn/Handles.h"
#include "torch/csrc/cudnn/Types.h"
extern THCState* state;
using namespace torch::cudnn;
#endif

using namespace torch::nn;
using thpp::Tensor;
using torch::cudnn::Convolution;
using tensor_pair = std::pair<std::unique_ptr<Tensor>, std::unique_ptr<Tensor>>;

namespace torch { namespace autograd {

// Forward function definition and utility functions

static std::unique_ptr<Tensor> compute_output(
  Tensor* input, Tensor* weight, Tensor* bias, Tensor* columns, Tensor* ones,
  const std::vector<long>& kernel_size, const ConvParams& params);

static std::unique_ptr<Tensor> compute_grad_input(
  Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* columns, Tensor* ones,
  const std::vector<long>& kernel_size, const ConvParams& params);

static tensor_pair compute_grad_params(
  Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* bias, Tensor* columns, Tensor* ones,
  const std::vector<long>& kernel_size, const ConvBackward& params);

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

auto ConvParams::use_cudnn(const Tensor& input) const -> bool {
#ifdef WITH_CUDNN
  if (!input.isCuda() || !cudnn_enabled) {
    return false;
  }
  if (is_dilated()) {
    cudaDeviceProp* prop = THCState_getCurrentDeviceProperties(state);
    // NOTE: extra parenthesis around numbers disable clang warnings about dead code
    return ((CUDNN_VERSION >= (6021)) || (CUDNN_VERSION >= (6000) && prop->major >= 5));
  }
  return true;
#endif
  return false;
}

auto ConvForward::output_size(Tensor& input, Tensor& weight) -> std::vector<long> {
  auto in_size = input.sizes();
  auto weight_size = weight.sizes();
  auto dim = input.nDim();

  std::vector<long> output_size(dim);
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

static auto view4d(const Tensor& tensor) -> std::unique_ptr<Tensor> {
  if (tensor.nDim() != 3) throw std::runtime_error("expected 3D tensor");
  auto result = tensor.newTensor();
  result->unsqueeze(tensor, 2);
  return result;
}

static auto view3d(const Tensor& tensor) -> std::unique_ptr<Tensor> {
  if (tensor.nDim() != 4) throw std::runtime_error("expected 4D tensor");
  auto result = tensor.newTensor();
  result->squeeze(tensor, 2);
  return result;
}

static std::unique_ptr<Tensor> subtensor(Tensor* tensor, int dim, int groups, int g) {
  if (!tensor) {
    return nullptr;
  }
  long n = tensor->rawSizes()[dim] / groups;
  auto result = tensor->newTensor();
  result->narrow(*tensor, dim, n * g, n);
  return result->contiguous();
}

static std::shared_ptr<Variable> subvariable(std::shared_ptr<Variable> var, int dim, int groups, int g) {
  long n = var->data->rawSizes()[dim] / groups;
  auto result = std::make_shared<Narrow>(dim, n * g, n)->apply({var})[0];
  return result;
}

static std::unique_ptr<Tensor> cat(const tensor_list& tensors, int dim) {
  int num_inputs = tensors.size();
  if (num_inputs == 0) {
    return nullptr;
  }

  std::vector<Tensor*> ptrs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ptrs[i] = tensors[i].get();
  }
  auto output = tensors[0]->newTensor();
  output->cat(ptrs, dim);
  return output;
}


// ConvForward implementation

auto ConvForward::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("ConvNd", inputs, 3, 2);
  if (is_padding_neg()) throw std::runtime_error("negative padding is not supported");
  if (is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");
  AutoGPU guard(inputs[0]->data->getDevice());
  auto input = inputs[0]->data->contiguous();
  std::unique_ptr<Tensor> weight(inputs[1]->data->clone_shallow());
  std::unique_ptr<Tensor> bias(inputs[2] ? inputs[2]->data->clone_shallow() : nullptr);

  int k = input->nDim();
  if (k == 3) {
    view1d_as_2d();
    input = view4d(*input);
    weight = view4d(*weight);
  }

  auto weight_size = weight->sizes();
  std::vector<long> kernel_size(weight_size.begin() + 2, weight_size.end());

  std::unique_ptr<Tensor> output;
  tensor_list columns(groups);
  tensor_list ones(groups);
  std::unique_ptr<Convolution> convolution;

  if (use_cudnn(*input)) {
#ifdef WITH_CUDNN
    if (input->type() != weight->type()){
      std::stringstream ss;
      ss << "Input type (" << thpp::toString(input->type()) << ") and weight type (" << thpp::toString(weight->type()) << ") should be the same";
      throw std::runtime_error(ss.str());
    }
    if (bias.get() != NULL && input->type() != bias->type()){
      std::stringstream ss;
      ss << "Input type (" << thpp::toString(input->type()) << ") and bias type (" << thpp::toString(bias->type()) << ") should be the same";
      throw std::runtime_error(ss.str());
    }
    output = input->newTensor();
    output->resize(output_size(*input, *weight));
    if (transposed) {
      convolution.reset(cudnn_convolution_transpose_full_forward(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
          (THVoidTensor*)input->cdata(), (THVoidTensor*)weight->cdata(),
          bias ? (THVoidTensor*)bias->cdata() : nullptr, (THVoidTensor*)output->cdata(),
          padding, stride, dilation, groups, benchmark));
    } else {
      convolution.reset(cudnn_convolution_full_forward(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
          (THVoidTensor*)input->cdata(), (THVoidTensor*)weight->cdata(),
          bias ? (THVoidTensor*)bias->cdata() : nullptr, (THVoidTensor*)output->cdata(),
          padding, stride, dilation, groups, benchmark));
    }
#endif
  } else {
    for (int g = 0; g < groups; ++g) {
      columns[g] = input->newTensor();
      ones[g] = input->newTensor();
    }
    if (groups == 1) {
      output = compute_output(
          input.get(), weight.get(), bias.get(),
          columns[0].get(), ones[0].get(), kernel_size, *this);
    } else {
      tensor_list outputs(groups);
      for (int g = 0; g < groups; ++g) {
        auto input_g = subtensor(input.get(), 1, groups, g);
        auto weight_g = subtensor(weight.get(), 0, groups, g);
        auto bias_g = subtensor(bias.get(), 0, groups, g);
        outputs[g] = compute_output(
            input_g.get(), weight_g.get(), bias_g.get(),
            columns[g].get(), ones[g].get(), kernel_size, *this);
      }
      output = cat(outputs, 1);
    }
  }

  if (k == 3) {
    output = view3d(*output);
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

  std::unique_ptr<thpp::Tensor> input {input_var->data->clone_shallow()};
  std::unique_ptr<thpp::Tensor> weight {weight_var->data->clone_shallow()};
  std::unique_ptr<thpp::Tensor> bias {bias_var ? bias_var->data->clone_shallow() : nullptr};

  AutoGPU guard(input->getDevice());

  input = input->contiguous();
  auto grad_output = grad_outputs[0]->data->contiguous();

  int k = input->nDim();
  if (k == 3) {
    input = view4d(*input);
    weight = view4d(*weight);
    grad_output = view4d(*grad_output);
  }

  auto weight_size = weight->sizes();
  std::vector<long> kernel_size(weight_size.begin() + 2, weight_size.end());

  bool use_cudnn = this->use_cudnn(*input);

  std::unique_ptr<Tensor> grad_input;
  std::unique_ptr<Tensor> grad_weight;
  std::unique_ptr<Tensor> grad_bias;

  if (should_compute_output(0)) {
    if (use_cudnn) {
#ifdef WITH_CUDNN
      grad_input = input->newTensor();
      grad_input->resizeAs(*input);
      if (transposed) {
        // ConvTranspose uses the same kernels as regular convolution
        // but swaps forward and backward calls
        cudnn_convolution_forward(
            state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
            (THVoidTensor*)grad_output->cdata(), (THVoidTensor*)weight->cdata(), (THVoidTensor*)grad_input->cdata(),
            convolution.get(), benchmark);
      } else {
        cudnn_convolution_backward_data(
            state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
            (THVoidTensor*)grad_output->cdata(), (THVoidTensor*)grad_input->cdata(), (THVoidTensor*)weight->cdata(),
            convolution.get(), benchmark);
      }
#endif
    } else if (groups == 1) {
      grad_input = compute_grad_input(
          input.get(), grad_output.get(), weight.get(),
          columns[0].get(), ones[0].get(), kernel_size, *this);
    } else {
      tensor_list grad_inputs(groups);
      for (int g = 0; g < groups; ++g) {
        auto input_g = subtensor(input.get(), 1, groups, g);
        auto grad_output_g = subtensor(grad_output.get(), 1, groups, g);
        auto weight_g = subtensor(weight.get(), 0, groups, g);
        grad_inputs[g] = compute_grad_input(
            input_g.get(), grad_output_g.get(), weight_g.get(),
            columns[g].get(), ones[g].get(), kernel_size, *this);
      }
      grad_input = cat(grad_inputs, 1);
    }
  }

  if (should_compute_output(1) || should_compute_output(2)) {
    if (use_cudnn) {
#ifdef WITH_CUDNN
      grad_weight = weight->newTensor();
      grad_weight->resizeAs(*weight);
      cudnn_convolution_backward_filter(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
          (THVoidTensor*)grad_output->cdata(), (THVoidTensor*)input->cdata(), (THVoidTensor*)grad_weight->cdata(),
          convolution.get(), benchmark);

      if (bias && should_compute_output(2)) {
        grad_bias = bias->newTensor();
        grad_bias->resizeAs(*bias);
        cudnn_convolution_backward_bias(
            state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
            (THVoidTensor*)grad_output->cdata(), (THVoidTensor*)grad_bias->cdata(),
            convolution.get());
      }
#endif
    } else if (groups == 1) {
      std::tie(grad_weight, grad_bias) = compute_grad_params(
          input.get(), grad_output.get(), weight.get(), bias.get(),
          columns[0].get(), ones[0].get(), kernel_size, *this);
    } else {
      tensor_list grad_weights(groups);
      tensor_list grad_biases(groups);
      for (int g = 0; g < groups; ++g) {
        auto input_g = subtensor(input.get(), 1, groups, g);
        auto grad_output_g = subtensor(grad_output.get(), 1, groups, g);
        auto weight_g = subtensor(weight.get(), 0, groups, g);
        auto bias_g = subtensor(bias.get(), 0, groups, g);
        std::tie(grad_weights[g], grad_biases[g]) = compute_grad_params(
            input_g.get(), grad_output_g.get(), weight_g.get(), bias_g.get(),
            columns[g].get(), ones[g].get(), kernel_size, *this);
      }
      grad_weight = cat(grad_weights, 0);
      if (bias && should_compute_output(2)) {
        grad_bias = cat(grad_biases, 0);
      }
    }
  }

  if (k == 3) {
    if (grad_input) {
      grad_input = view3d(*grad_input);
    }
    if (grad_weight) {
      grad_weight = view3d(*grad_weight);
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
  auto input = input_.unpack();

  // Compute ggO = conv(w, ggI) + conv(ggW, i) + ggb
  std::shared_ptr<Variable> ggO = nullptr;
  if (ggI) {
    if (weight->data->isCuda()) {
      weight = Contiguous().apply({weight})[0];
    }
    ggO = ConvForward(*this).apply({ggI, weight, nullptr})[0];
  }

  if (ggW) {
    if (ggW->data->isCuda()) {
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
    std::vector<long> new_size(gO->data->nDim(), 1);
    new_size[1] = ggb->data->rawSizes()[0];
    auto ggb_contiguous = Contiguous().apply({ggb})[0];
    auto ggb_view = View(new_size).apply({ggb_contiguous})[0];

    // Expand 
    auto ggb_expanded = Expand(gO->data->sizes()).apply({ggb_view})[0];

    if (ggO) {
      ggO = Add().apply({ggO, ggb_expanded})[0];
    } else {
      ggO = ggb_expanded;
    }
  }

  // Compute gW = conv(ggI, gO)
  std::shared_ptr<Variable> gW = nullptr;
  if (ggI) {
    // Modified params with correct padding
    ConvParams gw_conv_params(*this);

    // Disable groups as they are handled separately
    auto groups = gw_conv_params.groups;
    gw_conv_params.groups = 1;

    std::swap(gw_conv_params.dilation, gw_conv_params.stride);

    // Transpose gO and ggI to accumulate over batch
    auto gOt = Transpose(0, 1).apply({gO})[0];
    auto ggIt = Transpose(0, 1).apply({ggI})[0];

    std::shared_ptr<Variable> gWt = nullptr;
    // Compute conv
    if (groups == 1) {
      if (gOt->data->isCuda()) {
        gOt = Contiguous().apply({gOt})[0];
      }

      // Compute conv
      gWt = ConvForward(gw_conv_params).apply({ggIt, gOt, nullptr})[0];
    } else {
      variable_list gWt_list(groups);
      for (int g = 0; g < groups; ++g) {
        auto ggIt_g = subvariable(ggIt, 0, groups, g);
        auto gOt_g = subvariable(gOt, 0, groups, g);
        if (gOt_g->data->isCuda()) {
          gOt_g = Contiguous().apply({gOt_g})[0];
        }

        gWt_list[g] = ConvForward(gw_conv_params).apply({ggIt_g, gOt_g, nullptr})[0];
      }

      gWt = Cat(1).apply(gWt_list)[0];
    }

    // Transpose gW to match chan_in and chan_out
    gW = Transpose(0, 1).apply({gWt})[0];

    // narrow gW to only relevant portion
    // we do it this way instead of narrowing the input itself because
    // the ConvForward kernels don't support asymmetric padding.
    auto gW_size = gW->data->sizes();
    auto w_size = weight->data->sizes();
    for(size_t i = 2; i < gW_size.size(); ++i) {
      if (gW_size[i] > w_size[i]) {
        gW = Narrow(i, 0, w_size[i]).apply({gW})[0];
      }
    }
  }

  // Compute gI = convT(gO, ggW)
  std::shared_ptr<Variable> gI = nullptr;
  if (ggW) {
    // select conv tranpose
    ConvParams gi_conv_params(*this);
    gi_conv_params.transposed = true;

    // swap stride and dilation
    std::swap(gi_conv_params.dilation, gi_conv_params.stride);

    // calculate output_padding
    auto weight_size = weight->data->sizes();
    std::vector<long> kernel_size(weight_size.begin() + 2, weight_size.end());
    auto input_size = input->data->sizes();
    std::vector<long> input_shape(input_size.begin() + 2, input_size.end());
    auto grad_output_size = gO->data->sizes();
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

    auto ggWt = Transpose(0, 1).apply({ggW})[0];
    auto gOt = Transpose(0, 1).apply({gO})[0];

    std::shared_ptr<Variable> gIt = nullptr;
    if (groups == 1) {
      if (gOt->data->isCuda()) {
        gOt = Contiguous().apply({gOt})[0];
      }

      gIt = ConvForward(gi_conv_params).apply({ggWt, gOt, nullptr})[0];
    } else {
      variable_list gIt_list(groups);
      for (int g = 0; g < groups; ++g) {
        auto ggWt_g = subvariable(ggWt, 1, groups, g);
        auto gOt_g = subvariable(gOt, 0, groups, g);
        if (gOt_g->data->isCuda()) {
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

static std::unique_ptr<Tensor> compute_output(
    Tensor* input, Tensor* weight, Tensor* bias,
    Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size,
    const ConvParams& params) {

  auto output = input->newTensor();
  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (params.transposed) {
    if (dim == 4) {
      SpatialFullDilatedConvolution_updateOutput(
          input, output.get(), weight, bias, columns, ones,
          kernel_size[1], kernel_size[0],
          params.stride[1], params.stride[0],
          params.padding[1], params.padding[0],
          dilated ? params.dilation[1] : 1,
          dilated ? params.dilation[0] : 1,
          params.output_padding[1], params.output_padding[0]); goto done;
    } else if (dim == 5) {
      VolumetricFullDilatedConvolution_updateOutput(
          input, output.get(), weight, bias, columns, ones,
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
        SpatialDilatedConvolution_updateOutput(
          input, output.get(), weight, bias, columns, ones,
          kernel_size[1], kernel_size[0],
          params.stride[1], params.stride[0],
          params.padding[1], params.padding[0],
          params.dilation[1], params.dilation[0]); goto done;
      } else {
        /* CPU implementation has specialized MM kernels 
           for non-dilated case here */
        SpatialConvolutionMM_updateOutput(
            input, output.get(), weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
      }
    } else if (dim == 5 && (input->isCuda() || dilated)) {
      VolumetricDilatedConvolution_updateOutput(
          input, output.get(), weight, bias, columns, ones,
          kernel_size[0], kernel_size[2], kernel_size[1],
          params.stride[0], params.stride[2], params.stride[1],
          params.padding[0], params.padding[2], params.padding[1],
          dilated ? params.dilation[0] : 1,
          dilated ? params.dilation[2] : 1,
          dilated ? params.dilation[1] : 1); goto done;
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
      /* CPU implementation has specialized MM kernels 
         for non-dilated case here */
      VolumetricConvolutionMM_updateOutput(
          input, output.get(), weight, bias, columns,
          kernel_size[0], kernel_size[2], kernel_size[1],
          params.stride[0], params.stride[2], params.stride[1],
          params.padding[0], params.padding[2], params.padding[1]); goto done;
    }
  }
    
  throw std::runtime_error("unsupported ConvNd parameters");

done:
  return output;
}

static std::unique_ptr<Tensor> compute_grad_input(
    Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size, const ConvParams& params) {

  auto grad_input = input->newTensor();
  grad_input->resizeAs(*input);
  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (params.transposed) {
    if (dim == 4) {
      SpatialFullDilatedConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            dilated ? params.dilation[1] : 1,
            dilated ? params.dilation[0] : 1,
            params.output_padding[1], params.output_padding[0]); goto done;
    } else if (dim == 5) {
      VolumetricFullDilatedConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns, ones,
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
        SpatialDilatedConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0]); goto done;
      } else {
        /* CPU implementation has specialized MM kernels 
           for non-dilated case here */
        SpatialConvolutionMM_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
      }
    } else if (dim == 5 && (input->isCuda() || dilated)) {
        VolumetricDilatedConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            dilated ? params.dilation[0] : 1,
            dilated ? params.dilation[2] : 1,
            dilated ? params.dilation[1] : 1); goto done;
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU implementation has specialized MM kernels 
           for non-dilated case here */
        VolumetricConvolutionMM_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns, ones,
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
    Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* bias,
    Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size, const ConvBackward& params) {

  auto grad_weight = weight->newTensor();
  grad_weight->resizeAs(*weight).zero();

  std::unique_ptr<Tensor> grad_bias;
  if (bias && params.should_compute_output(2)) {
    grad_bias = bias->newTensor();
    grad_bias->resizeAs(*bias).zero();
  }

  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (params.transposed) {
    if (dim == 4) {
      SpatialFullDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            dilated ? params.dilation[1] : 1,
            dilated ? params.dilation[0] : 1,
            params.output_padding[1], params.output_padding[0], 1.0); goto done;
    } else if (dim == 5) {
        VolumetricFullDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
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
        SpatialDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0], 1.0); goto done;
      } else {
        /* CPU implementation has specialized MM kernels 
           for non-dilated case here */
        SpatialConvolutionMM_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0], 1.0); goto done;
      }
    } else if (dim == 5 && (input->isCuda() || dilated)) {
        VolumetricDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[0], kernel_size[2], kernel_size[1],
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            dilated ? params.dilation[0] : 1,
            dilated ? params.dilation[2] : 1,
            dilated ? params.dilation[1] : 1, 1.0); goto done;
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU implementation has specialized MM kernels 
           for non-dilated case here */
        VolumetricConvolutionMM_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns,
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

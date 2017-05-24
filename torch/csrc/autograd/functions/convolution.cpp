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
    return ((CUDNN_VERSION >= (6021)) || (CUDNN_VERSION >= (6000) && prop->major >= 5)) && !transposed;
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
    return std::unique_ptr<Tensor>();
  }
  long n = tensor->sizes()[dim] / groups;
  auto result = tensor->newTensor();
  result->narrow(*tensor, dim, n * g, n);
  return result->contiguous();
}

static std::unique_ptr<Tensor> cat(const tensor_list& tensors, int dim) {
  int num_inputs = tensors.size();
  if (num_inputs == 0) {
    return std::unique_ptr<Tensor>();
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
  auto input = input_.unpack_data();
  AutoGPU guard(input->getDevice());
  input = input->contiguous();
  std::unique_ptr<Tensor> weight(weight_.unpack_data()->clone_shallow());
  auto bias = bias_.unpack_data();
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
  variable_list all_grad_outputs(grad_outputs);
  all_grad_outputs.push_back(input_.unpack());
  all_grad_outputs.push_back(weight_.unpack());

  auto outputs =  as_tensor_list(std::move(grad_input),
                                 std::move(grad_weight),
                                 std::move(grad_bias));
  return wrap_outputs(all_grad_outputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<ConvBackwardBackward>(
      f, *this,
      input_.unpack()->save(this), weight_.unpack()->save(this),
      Variable::save_opt(bias_.unpack().get(), this), grad_outputs[0]->save(this));
  });
};

auto ConvBackward::releaseVariables() -> void {
  input_.data.reset();
  weight_.data.reset();
  bias_.data.reset();
}


// ConvBackwardBackward implementation

auto ConvBackwardBackward::apply(const variable_list& grad_grad_inputs) -> variable_list {
  if (grad_grad_inputs.size() != 3) throw std::runtime_error("expected three grad_grad_inputs");
  if (transposed) throw std::runtime_error("ConvBackwardBackward does not support transposed convolution");

  auto ggI = grad_grad_inputs[0];
  auto ggW = grad_grad_inputs[1];
  auto ggb = grad_grad_inputs[2];

  auto gO = grad_output_.unpack();
  auto weight = weight_.unpack();

  // Compute ggO = conv(w, ggI) + conv(ggW, i) + ggb
  std::shared_ptr<Variable> ggO = nullptr;
  if (ggI) {
    ggO = std::make_shared<ConvForward>(*this)->apply({ggI, weight, nullptr})[0];
  }

  if (ggW) {
    auto ggW_term = std::make_shared<ConvForward>(*this)->apply({input_.unpack(), ggW, nullptr})[0];
    if (ggO) {
      ggO = std::make_shared<Add>()->apply({ggO, ggW_term})[0];
    } else {
      ggO = ggW_term;
    }
  }

  if (ggb) {
    // View as (1, ggb.size(0), 1, 1...)
    std::vector<long> new_size(gO->data->sizes().size(), 1);
    new_size[1] = ggb->data->sizes()[0];
    auto ggb_contiguous = std::make_shared<Clone>()->apply({ggb})[0];
    auto ggb_view = std::make_shared<View>(new_size)->apply({ggb_contiguous})[0];

    // Expand 
    auto ggb_expanded = std::make_shared<Expand>(gO->data->sizes())->apply({ggb_view})[0];

    if (ggO) {
      ggO = std::make_shared<Add>()->apply({ggO, ggb_expanded})[0];
    } else {
      ggO = ggb_expanded;
    }
  }

  // Compute gW = conv(gO, ggI)
  std::shared_ptr<Variable> gW = nullptr;
  if (ggI) {
    // Modified params with correct padding
    ConvParams gw_conv_params(*this);
    auto weight_size = weight->data->sizes();
    std::vector<long> kernel_size(weight_size.begin() + 2, weight_size.end());
    auto input_size = ggI->data->sizes();
    std::vector<long> input_shape(input_size.begin() + 2, input_size.end());
    for(size_t i=0; i<gw_conv_params.padding.size(); ++i) {
      // Formula for conv output size before the floor operation
      auto out_size = float(input_shape[i] + 2 * gw_conv_params.padding[i] +
                      gw_conv_params.dilation[i] * (kernel_size[i] - 1) - 1) /
                      gw_conv_params.stride[i] + 1;
      if (floorf(out_size) != out_size) {
        // TODO: narrow ggI here to ignore these elements?
        throw std::runtime_error("Some input elements have been lost during ConvForward"
        " (see documentation for the Conv layer) so ConvBackwardBackward cannot be used."
        " Resize the input so that no element is lost to be able to use ConvBackwardBackward.");
      }
      auto tmp = gw_conv_params.dilation[i];
      gw_conv_params.dilation[i] = gw_conv_params.stride[i];
      gw_conv_params.stride[i] = tmp;
    }

    // Transpose gO and ggI to accumulate over batch
    auto gOt = std::make_shared<Transpose>(0, 1)->apply({gO})[0];
    auto ggIt = std::make_shared<Transpose>(0, 1)->apply({ggI})[0];

    // Compute conv
    auto gWt = std::make_shared<ConvForward>(gw_conv_params)->apply({ggIt, gOt, nullptr})[0];

    // Transpose gW to match chan_in and chan_out
    gW = std::make_shared<Transpose>(0, 1)->apply({gWt})[0];
  }

  // Compute gI = convT(gO, ggW)
  std::shared_ptr<Variable> gI = nullptr;
  if (ggW) {
    // select conv tranpose and swap stride and dilation
    ConvParams gi_conv_params(*this);
    gi_conv_params.transposed = true;
    for(size_t i=0; i<gi_conv_params.padding.size(); ++i) {
      if (gi_conv_params.stride[i] != 1) {
        // TODO: Remove this when transpose dilated is fixed
        throw std::runtime_error("Setting non-zero ggW for ConvBackwardBackward would require"
        " using a dilated transpose convolution which is not supported.");
      }
      auto tmp = gi_conv_params.dilation[i];
      gi_conv_params.dilation[i] = gi_conv_params.stride[i];
      gi_conv_params.stride[i] = tmp;
    }

    auto ggWt = std::make_shared<Transpose>(0, 1)->apply({ggW})[0];
    // Weight for conv transpose are (chan_in, chan_out, kern, kern)
    auto gOt = std::make_shared<Transpose>(0, 1)->apply({gO})[0];

    auto gIt = std::make_shared<ConvForward>(gi_conv_params)->apply({ggWt, gOt, nullptr})[0];
    
    gI = std::make_shared<Transpose>(0, 1)->apply({gIt})[0];
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

  if (dilated) {
    if (params.transposed) {
      /* dilated && transposed */
      /* NOT IMPLEMENTED */
    } else /* !transposed */ {
      /* dilated && !transposed */
      if (dim == 4) {
        SpatialDilatedConvolution_updateOutput(
            input, output.get(), weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0]); goto done;
      } else if (dim == 5) {
        VolumetricDilatedConvolution_updateOutput(
            input, output.get(), weight, bias, columns, ones,
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
        SpatialFullConvolution_updateOutput(
            input, output.get(), weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.output_padding[1], params.output_padding[0]); goto done;
      } else if (dim == 5) {
        VolumetricFullConvolution_updateOutput(
            input, output.get(), weight, bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.output_padding[0], params.output_padding[2], params.output_padding[1]); goto done;
      }
    } else /* !transposed */ {
      /* !dilated && !transposed */
      if (dim == 4) {
        SpatialConvolutionMM_updateOutput(
            input, output.get(), weight, bias, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
      } else if (dim == 5 && input->isCuda()) {
        VolumetricConvolution_updateOutput(
            input, output.get(), weight, bias, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
      } else if (dim == 5) {
        VolumetricConvolutionMM_updateOutput(
            input, output.get(), weight, bias, columns,
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

static std::unique_ptr<Tensor> compute_grad_input(
    Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size, const ConvParams& params) {

  auto grad_input = input->newTensor();
  grad_input->resizeAs(*input);
  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (dilated) {
    if (params.transposed) {
      /* dilated && transposed */
      /* NOT IMPLEMENTED */
    } else /* !transposed */ {
      /* dilated && !transposed */
      if (dim == 4) {
        SpatialDilatedConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0]); goto done;
      } else if (dim == 5) {
        VolumetricDilatedConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
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
        SpatialFullConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.output_padding[1], params.output_padding[0]); goto done;
      } else if (dim == 5) {
        VolumetricFullConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.output_padding[0], params.output_padding[2], params.output_padding[1]); goto done;
      }
    } else /* !transposed */ {
      /* !dilated && !transposed */
      if (dim == 4) {
        SpatialConvolutionMM_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0]); goto done;
      } else if (dim == 5 && input->isCuda()) {
        VolumetricConvolution_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1]); goto done;
      } else if (dim == 5) {
        VolumetricConvolutionMM_updateGradInput(
            input, grad_output, grad_input.get(), weight, columns, ones,
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

  if (dilated) {
    if (params.transposed) {
      /* dilated && transposed */
      /* NOT IMPLEMENTED */
    } else /* !transposed */ {
      /* dilated && !transposed */
      if (dim == 4) {
        SpatialDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.dilation[1], params.dilation[0], 1.0); goto done;
      } else if (dim == 5) {
        VolumetricDilatedConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
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
        SpatialFullConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0],
            params.output_padding[1], params.output_padding[0], 1.0); goto done;
      } else if (dim == 5) {
        VolumetricFullConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1],
            params.output_padding[0], params.output_padding[2], params.output_padding[1], 1.0); goto done;
      }
    } else /* !transposed */ {
      /* !dilated && !transposed */
      if (dim == 4) {
        SpatialConvolutionMM_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            kernel_size[1], kernel_size[0],
            params.stride[1], params.stride[0],
            params.padding[1], params.padding[0], 1.0); goto done;
      } else if (dim == 5 && input->isCuda()) {
        VolumetricConvolution_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
            params.stride[0], params.stride[2], params.stride[1],
            params.padding[0], params.padding[2], params.padding[1], 1.0); goto done;
      } else if (dim == 5) {
        VolumetricConvolutionMM_accGradParameters(
            input, grad_output, grad_weight.get(), grad_bias.get(), columns,
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


// Utils functions
}} // namespace torch::autograd

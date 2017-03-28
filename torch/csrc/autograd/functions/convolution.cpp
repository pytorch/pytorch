#include "convolution.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/nn/THNN_generic.h"
#include "torch/csrc/utils/auto_gpu.h"

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

static std::unique_ptr<Tensor> subtensor(Tensor* tensor, int dim, int groups, int g);

static std::unique_ptr<Tensor> compute_output(
  Tensor* input, Tensor* weight, Tensor* bias, Tensor* columns, Tensor* ones,
  const std::vector<long>& kernel_size, const ConvParams& params);

static std::unique_ptr<Tensor> compute_grad_input(
  Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* columns, Tensor* ones,
  const std::vector<long>& kernel_size, const ConvParams& params);

static tensor_pair compute_grad_params(
  Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* bias, Tensor* columns, Tensor* ones,
  const std::vector<long>& kernel_size, const ConvBackward& params);

static std::unique_ptr<Tensor> cat(const tensor_list& tensors, int dim);

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

auto ConvForward::apply(const variable_list& inputs) -> variable_list {
  if (inputs.size() != 3) throw std::runtime_error("expected three inputs");
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

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->isCuda() && (!is_dilated() || CUDNN_VERSION >= 6000)) && cudnn_enabled;
#endif

  std::unique_ptr<Tensor> output;
  tensor_list columns(groups);
  tensor_list ones(groups);
  std::unique_ptr<Convolution> convolution;

  if (use_cudnn) {
#ifdef WITH_CUDNN
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

  auto creator = std::make_shared<ConvBackward>(
      flags(inputs), *this,
      inputs[0]->save(), inputs[1]->save(), Variable::save_opt(inputs[2].get()),
      std::move(columns), std::move(ones), std::move(convolution));

  variable_list results(1);
  results[0] = std::make_shared<Variable>(std::move(output), creator);
  return results;
};

auto ConvBackward::apply(const variable_list& grad_outputs) -> variable_list {
  if (grad_outputs.size() != 1) throw std::runtime_error("expected one grad_output");
  if (is_padding_neg()) throw std::runtime_error("negative padding is not supported");
  if (is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");

  AutoGPU guard(input_.data->getDevice());

  auto input = input_.unpack()->contiguous();
  std::unique_ptr<Tensor> weight(weight_.unpack()->clone_shallow());
  std::unique_ptr<Tensor> bias(bias_.unpack() ? bias_.unpack()->clone_shallow() : nullptr);
  auto grad_output = grad_outputs[0]->data->contiguous();

  int k = input->nDim();
  if (k == 3) {
    input = view4d(*input);
    weight = view4d(*weight);
    grad_output = view4d(*grad_output);
  }

  auto weight_size = weight->sizes();
  std::vector<long> kernel_size(weight_size.begin() + 2, weight_size.end());

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->isCuda() && (!is_dilated() || CUDNN_VERSION >= 6000)) && cudnn_enabled;
#endif

  std::unique_ptr<Tensor> grad_input;
  std::unique_ptr<Tensor> grad_weight;
  std::unique_ptr<Tensor> grad_bias;

  if (needs_input_grad(0)) {
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

  if (needs_input_grad(1) || needs_input_grad(2)) {
    if (use_cudnn) {
#ifdef WITH_CUDNN
      grad_weight = weight->newTensor();
      grad_weight->resizeAs(*weight);
      cudnn_convolution_backward_filter(
          state, torch::cudnn::getCudnnHandle(), torch::cudnn::getCudnnDataType(*input),
          (THVoidTensor*)grad_output->cdata(), (THVoidTensor*)input->cdata(), (THVoidTensor*)grad_weight->cdata(),
          convolution.get(), benchmark);

      if (bias && needs_input_grad(2)) {
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
      grad_bias = cat(grad_biases, 0);
    }
  }

  if (k == 3) {
    if (needs_input_grad(0)) {
        grad_input = view3d(*grad_input);
    }
    grad_weight = view3d(*grad_weight);
  }

  variable_list results(3);
  results[0] = Variable::of(std::move(grad_input));
  results[1] = Variable::of(std::move(grad_weight));
  results[2] = Variable::of(std::move(grad_bias));
  return results;
};

auto ConvBackward::releaseVariables() -> void {
  input_.data.reset();
  weight_.data.reset();
  bias_.data.reset();
}

static std::unique_ptr<Tensor> compute_output(
    Tensor* input, Tensor* weight, Tensor* bias,
    Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size,
    const ConvParams& params) {

  auto output = input->newTensor();
  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (params.transposed && dim == 4) {
    SpatialFullConvolution_updateOutput(
        input, output.get(), weight, bias, columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0],
        params.output_padding[1], params.output_padding[0]);
  } else if (params.transposed && dim == 5) {
    VolumetricFullConvolution_updateOutput(
        input, output.get(), weight, bias, columns, ones,
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1],
        params.output_padding[0], params.output_padding[2], params.output_padding[1]);
  } else if (dilated && dim == 4) {
    SpatialDilatedConvolution_updateOutput(
        input, output.get(), weight, bias, columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0],
        params.dilation[1], params.dilation[0]);
  } else if (dilated && dim == 5) {
    VolumetricDilatedConvolution_updateOutput(
        input, output.get(), weight, bias, columns, ones,
        kernel_size[0], kernel_size[2], kernel_size[1],
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1],
        params.dilation[0], params.dilation[2], params.dilation[1]);
  } else if (dim == 4) {
    SpatialConvolutionMM_updateOutput(
        input, output.get(), weight, bias, columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0]);
  } else if (dim == 5 && input->isCuda()) {
    VolumetricConvolution_updateOutput(
        input, output.get(), weight, bias, columns, ones,
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1]);
  } else if (dim == 5) {
    VolumetricConvolutionMM_updateOutput(
        input, output.get(), weight, bias, columns,
        kernel_size[0], kernel_size[2], kernel_size[1],
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1]);
  }
  return output;
}

static std::unique_ptr<Tensor> compute_grad_input(
    Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size, const ConvParams& params) {

  auto grad_input = input->newTensor();
  grad_input->resizeAs(*input);
  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (params.transposed && dim == 4) {
    SpatialFullConvolution_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0],
        params.output_padding[1], params.output_padding[0]);
  } else if (params.transposed && dim == 5) {
    VolumetricFullConvolution_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns, ones,
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1],
        params.output_padding[0], params.output_padding[2], params.output_padding[1]);
  } else if (dilated && dim == 4) {
    SpatialDilatedConvolution_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0],
        params.dilation[1], params.dilation[0]);
  } else if (dilated && dim == 5) {
    VolumetricDilatedConvolution_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns,
        kernel_size[0], kernel_size[2], kernel_size[1],
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1],
        params.dilation[0], params.dilation[2], params.dilation[1]);
  } else if (dim == 4) {
    SpatialConvolutionMM_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0]);
  } else if (dim == 5 && input->isCuda()) {
    VolumetricConvolution_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns,
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1]);
  } else if (dim == 5) {
    VolumetricConvolutionMM_updateGradInput(
        input, grad_output, grad_input.get(), weight, columns, ones,
        kernel_size[0], kernel_size[2], kernel_size[1],
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1]);
  }
  return grad_input;
}

static tensor_pair compute_grad_params(
    Tensor* input, Tensor* grad_output, Tensor* weight, Tensor* bias,
    Tensor* columns, Tensor* ones,
    const std::vector<long>& kernel_size, const ConvBackward& params) {

  auto grad_weight = weight->newTensor();
  grad_weight->resizeAs(*weight).zero();

  std::unique_ptr<Tensor> grad_bias;
  if (bias && params.needs_input_grad(2)) {
    grad_bias = bias->newTensor();
    grad_bias->resizeAs(*bias).zero();
  }

  auto dim = input->nDim();
  auto dilated = params.is_dilated();

  if (params.transposed && dim == 4) {
    SpatialFullConvolution_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0],
        params.output_padding[1], params.output_padding[0], 1.0);
  } else if (params.transposed && dim == 5) {
    VolumetricFullConvolution_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1],
        params.output_padding[0], params.output_padding[2], params.output_padding[1], 1.0);
  } else if (dilated && dim == 4) {
    SpatialDilatedConvolution_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0],
        params.dilation[1], params.dilation[0], 1.0);
  } else if (dilated && dim == 5) {
    VolumetricDilatedConvolution_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
        kernel_size[0], kernel_size[2], kernel_size[1],
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1],
        params.dilation[0], params.dilation[2], params.dilation[1], 1.0);
  } else if (dim == 4) {
    SpatialConvolutionMM_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
        kernel_size[1], kernel_size[0],
        params.stride[1], params.stride[0],
        params.padding[1], params.padding[0], 1.0);
  } else if (dim == 5 && input->isCuda()) {
    VolumetricConvolution_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns, ones,
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1], 1.0);
  } else if (dim == 5) {
    VolumetricConvolutionMM_accGradParameters(
        input, grad_output, grad_weight.get(), grad_bias.get(), columns,
        kernel_size[0], kernel_size[2], kernel_size[1],
        params.stride[0], params.stride[2], params.stride[1],
        params.padding[0], params.padding[2], params.padding[1], 1.0);
  }
  return std::make_pair<>(std::move(grad_weight), std::move(grad_bias));
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

}} // namespace torch::autograd

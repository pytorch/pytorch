#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/hipdnn_batch_norm_native.h>
#include <ATen/ops/hipdnn_batch_norm_backward_native.h>
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at::native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  TORCH_CHECK(false, "hipdnn_batch_norm: ATen not compiled with ROCM support");
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    double epsilon) {
  TORCH_CHECK(false, "hipdnn_batch_norm_backward: ATen not compiled with ROCM support");
}

}  // namespace at::native

#elif !defined(USE_HIPDNN) // AT_ROCM_ENABLED but no hipDNN

namespace at::native {

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  TORCH_CHECK(false, "hipdnn_batch_norm: not compiled with hipDNN support");
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    double epsilon) {
  TORCH_CHECK(false, "hipdnn_batch_norm_backward: not compiled with hipDNN support");
}

}  // namespace at::native

#else // AT_ROCM_ENABLED && USE_HIPDNN

#include <hipdnn_frontend.hpp>
#include <ATen/hipdnn/Types.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/Exceptions.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{ 1, t.numel() };
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

}  // namespace

inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    createTensorAttributes(const Tensor& t)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    tensor->set_dim(t.sizes().vec()).set_data_type(getHipdnnDataType(t));
    tensor->set_stride(t.strides().vec());
    return tensor;
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = running_mean_t_opt.value_or(Tensor());
  const Tensor& running_var_t = running_var_t_opt.value_or(Tensor());

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "hipdnn_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() == ScalarType::Half || input->scalar_type() == ScalarType::BFloat16) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  auto output_t = at::empty_like(input_t, input_t.options(), input_t.suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getHipdnnHandle();
  auto inputType = getHipdnnDataType(*input);
  auto intermediateType = getHipdnnDataType(*weight);
  Tensor save_mean, save_var;

  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto input_attr = createTensorAttributes(*input);
    auto weight_attr = createTensorAttributes(expandScale(*weight, input->dim()));
    auto bias_attr = createTensorAttributes(expandScale(*bias, input->dim()));

    auto bnAttributes = hipdnn_frontend::graph::BatchnormAttributes();
    auto epsilon_attr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon_attr->set_value(epsilon);
    bnAttributes.set_epsilon(epsilon_attr);

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prev_mean_attr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prev_var_attr;

    if (running_mean->defined()) {
      prev_mean_attr = createTensorAttributes(expandScale(*running_mean, input->dim()));
      prev_var_attr = createTensorAttributes(expandScale(*running_var, input->dim()));
      auto momentum_attr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
      momentum_attr->set_value(exponential_average_factor);
      bnAttributes.set_previous_running_stats(prev_mean_attr, prev_var_attr, momentum_attr);
    }

    auto [y, savedMean, savedInvVar, nextMean, nextVar] =
        graph->batchnorm(input_attr, weight_attr, bias_attr, bnAttributes);
    y->set_output(true);
    savedMean->set_output(true).set_data_type(intermediateType);
    savedInvVar->set_output(true).set_data_type(intermediateType);
    if (running_mean->defined()) {
      nextMean->set_output(true).set_data_type(intermediateType);
      nextVar->set_output(true).set_data_type(intermediateType);
    }

    HIPDNN_FE_CHECK(graph->build(handle));

    int64_t workspace_size = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspace_size));
    auto workspace = at::empty({workspace_size}, input_t.options().dtype(at::kByte));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[input_attr->get_uid()] = input->data_ptr();
    variantPack[weight_attr->get_uid()] = weight->data_ptr();
    variantPack[bias_attr->get_uid()] = bias->data_ptr();
    variantPack[y->get_uid()] = output->data_ptr();
    variantPack[savedMean->get_uid()] = save_mean.data_ptr();
    variantPack[savedInvVar->get_uid()] = save_var.data_ptr();
    if (running_mean->defined()) {
      variantPack[prev_mean_attr->get_uid()] = running_mean->data_ptr();
      variantPack[prev_var_attr->get_uid()] = running_var->data_ptr();
      variantPack[nextMean->get_uid()] = running_mean->data_ptr();
      variantPack[nextVar->get_uid()] = running_var->data_ptr();
    }

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.data_ptr()));

  } else {
    save_mean = at::empty({0}, weight_t.options());
    save_var = at::empty({0}, weight_t.options());

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto input_attr = createTensorAttributes(*input);
    auto weight_attr = createTensorAttributes(expandScale(*weight, input->dim()));
    auto bias_attr = createTensorAttributes(expandScale(*bias, input->dim()));
    auto mean_attr = createTensorAttributes(expandScale(*running_mean, input->dim()));
    auto variance_attr = createTensorAttributes(expandScale(*running_var, input->dim()));
    auto epsilon_attr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon_attr->set_value(epsilon);

    auto bnAttributes = hipdnn_frontend::graph::BatchnormInferenceAttributesVarianceExt();
    auto output_attr = graph->batchnorm_inference_variance_ext(
        input_attr, mean_attr, variance_attr, weight_attr, bias_attr, epsilon_attr, bnAttributes);
    output_attr->set_output(true);

    HIPDNN_FE_CHECK(graph->build(handle));

    int64_t workspace_size = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspace_size));
    auto workspace = at::empty({workspace_size}, input_t.options().dtype(at::kByte));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[input_attr->get_uid()] = input->data_ptr();
    variantPack[weight_attr->get_uid()] = weight->data_ptr();
    variantPack[bias_attr->get_uid()] = bias->data_ptr();
    variantPack[mean_attr->get_uid()] = running_mean->data_ptr();
    variantPack[variance_attr->get_uid()] = running_var->data_ptr();
    variantPack[output_attr->get_uid()] = output->data_ptr();

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.data_ptr()));
  }

  return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm_backward(
    const Tensor& input_t,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_t_opt,
    const std::optional<Tensor>& save_var_t_opt,
    double epsilon) {

  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& save_mean_t = save_mean_t_opt.value_or(Tensor());
  const Tensor& save_var_t = save_var_t_opt.value_or(Tensor());

  auto grad_output_contig =
      grad_output_t.contiguous(input_t.suggest_memory_format());
  TensorArg input{input_t, "input", 1},
      grad_output{grad_output_contig, "grad_output", 2},
      weight{weight_t, "weight", 3}, save_mean{save_mean_t, "save_mean", 4},
      save_var{save_var_t, "save_var", 5};
  CheckedFrom c = "hipdnn_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half || input->scalar_type() == ScalarType::BFloat16) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {save_mean, save_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  TORCH_CHECK(grad_output->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  auto grad_input_t  = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());

  auto handle = getHipdnnHandle();
  auto inputType = getHipdnnDataType(*input);
  auto intermediateType = getHipdnnDataType(*weight);

  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType)
      .set_intermediate_data_type(intermediateType)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto dy_attr = createTensorAttributes(*grad_output);
  auto input_attr = createTensorAttributes(*input);
  auto weight_attr = createTensorAttributes(expandScale(*weight, input->dim()));
  auto savedMeanAttr = createTensorAttributes(expandScale(*save_mean, input->dim()));
  auto savedInvVarAttr = createTensorAttributes(expandScale(*save_var, input->dim()));

  auto bnBwdAttributes = hipdnn_frontend::graph::BatchnormBackwardAttributes();
  bnBwdAttributes.set_saved_mean_and_inv_variance(savedMeanAttr, savedInvVarAttr);

  auto [dx, dscale, dbias] = graph->batchnorm_backward(
      dy_attr, input_attr, weight_attr, bnBwdAttributes);
  dx->set_output(true);
  dscale->set_output(true).set_data_type(intermediateType);
  dbias->set_output(true).set_data_type(intermediateType);

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t workspace_size = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(workspace_size));
  auto workspace = at::empty({workspace_size}, input_t.options().dtype(at::kByte));

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[dy_attr->get_uid()] = grad_output->data_ptr();
  variantPack[input_attr->get_uid()] = input->data_ptr();
  variantPack[weight_attr->get_uid()] = weight->data_ptr();
  variantPack[savedMeanAttr->get_uid()] = save_mean->data_ptr();
  variantPack[savedInvVarAttr->get_uid()] = save_var->data_ptr();
  variantPack[dx->get_uid()] = grad_input_t.data_ptr();
  variantPack[dscale->get_uid()] = grad_weight_t.data_ptr();
  variantPack[dbias->get_uid()] = grad_bias_t.data_ptr();

  HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.data_ptr()));

  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif

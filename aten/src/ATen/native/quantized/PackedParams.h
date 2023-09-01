#pragma once
#include <map>

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;

  // out variant of LinearPackedParamsBase::apply
  virtual at::Tensor& apply_out(
      const at::Tensor& /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual at::Tensor& apply_relu_out(
      const at::Tensor& /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  // Corresponding pattern (the ops with `*` are part of the pattern that
  // represents the computation of quantized::linear_with_input_q_dq_qweight_dq_output_fp32):
  // input -> q* -> dq* -> linear* ->
  //         qweight -> dq* /
  //
  // After fusion:
  // input -> quantized::linear_with_input_q_dq_qweight_dq_output_fp32* ->
  //         qweight /
  //
  // Additional Note: the weight is packed as well
  // Params:
  //    X: float32 Tensor, will be quantized to quint8 in the op
  //    W_prepack: packed qint8 quantized weight and bias
  // Returns:
  //    Y: float32 Tensor
  virtual at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) {
    throw std::runtime_error(
        "apply_with_input_q_dq_qweight_dq_output_fp32 is not implemented for this packed "
        "parameter type");
    return {};
  }

  // Corresponding pattern (the ops with `*` are part of the pattern that
  // represents the computation of quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32):
  // input -> q* -> dq* -> linear* -> relu* ->
  //         qweight -> dq* /
  //
  // After fusion:
  // input -> quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32* ->
  //         qweight /
  //
  // Additional Note: the weight is packed as well
  // Params:
  //    input: float32 Tensor, will be quantized to quint8 in the op
  // Returns:
  //    float32 Tensor
  virtual at::Tensor apply_with_input_q_dq_qweight_dq_relu_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) {
    throw std::runtime_error(
        "apply_with_input_q_dq_qweight_dq_relu_output_fp32 is not implemented for this packed "
        "parameter type");
    return {};
  }

  virtual at::Tensor apply_dynamic(
      at::Tensor input,
      bool reduce_range = false) = 0;
  virtual at::Tensor apply_dynamic_relu(
      at::Tensor input,
      bool reduce_range = false) = 0;

  virtual at::Tensor& apply_dynamic_out(
      const at::Tensor& /* input */,
      at::Tensor& output,
      bool /* reduce_range */) {
    throw std::runtime_error(
        "apply_dynamic_out is not implemented for this packed "
        "parameter type");
    return output;
  }
  virtual at::Tensor& apply_dynamic_relu_out(
      const at::Tensor& /* input */,
      at::Tensor& output,
      bool /* reduce_range */) {
    throw std::runtime_error(
        "apply_dynamic_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() = 0;

  virtual c10::optional<at::Tensor> bias() = 0;

  virtual void set_bias(c10::optional<at::Tensor> /*bias*/) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }
};

template <int kSpatialDim = 2>
struct ConvPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range) = 0;

  virtual std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() = 0;

  virtual torch::List<int64_t> stride() const = 0;
  virtual torch::List<int64_t> padding() const = 0;
  virtual torch::List<int64_t> output_padding() const = 0;
  virtual torch::List<int64_t> dilation() const = 0;
  virtual int64_t groups() const = 0;
  virtual bool transpose() const = 0;
};
template<uint32_t kSpatialDim>
using prepack_fn = c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> (*)(at::Tensor weight,
        c10::optional<at::Tensor>,
        torch::List<int64_t> ,
        torch::List<int64_t> ,
        torch::List<int64_t> ,
        torch::List<int64_t> ,
        int64_t ,
        bool );

template <int kSpatialDim>
TORCH_API std::map<at::QEngine, prepack_fn<kSpatialDim>>& get_prepack_fns();
template <int kSpatialDim>
TORCH_API void register_prepack(at::QEngine device, prepack_fn<kSpatialDim> prepack);
template <int kSpatialDim>
TORCH_API prepack_fn<kSpatialDim> get_device_prepack_fn(at::QEngine device);


using linear_prepack_fn = c10::intrusive_ptr<LinearPackedParamsBase> (*)(at::Tensor ,
      c10::optional<at::Tensor>);

TORCH_API void register_linear_prepack(at::QEngine device, linear_prepack_fn prepack);
TORCH_API void register_linear_prepack_fp16(at::QEngine device, linear_prepack_fn prepack);
TORCH_API linear_prepack_fn get_device_linear_prepack_fn(at::QEngine device);
TORCH_API linear_prepack_fn get_device_linear_prepack_fn_fp16(at::QEngine device);

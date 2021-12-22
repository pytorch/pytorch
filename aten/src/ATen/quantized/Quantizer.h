#pragma once

#include <c10/core/QScheme.h>
#include <c10/core/MemoryFormat.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>

#include <ATen/core/QuantizerBase.h>

#include <cmath>
#include <memory>

namespace at {

/**
 * UniformQuantizer is the parent class for all uniform quantizers.
 * These quantization scheme will map float value uniformly to
 * the quantized value. For example, affine quantizer is
 * the most commonly used scheme in this category.
 */
struct TORCH_API UniformQuantizer : public Quantizer {
  explicit UniformQuantizer(ScalarType scalar_type) : Quantizer(scalar_type) {}
};

/**
 * NonUniformQuantizer is the parent class for all non-uniform quantizers.
 * These quantization scheme may map float value non-uniformly to the quantized
 * value. K-means quantization is a representative example in this category.
 */
struct TORCH_API NonUniformQuantizer : public Quantizer {
  explicit NonUniformQuantizer(ScalarType scalar_type) : Quantizer(scalar_type) {}
};

// There is also StochasticQuantizer which is uniform but not affine

/**
 * PerTensorQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct TORCH_API PerTensorQuantizer : public UniformQuantizer {
  explicit PerTensorQuantizer(ScalarType scalar_type, double scale, int64_t zero_point, QScheme qscheme=kPerTensorAffine)
    : UniformQuantizer(scalar_type),
        scale_(scale),
        zero_point_(zero_point),
        qscheme_(qscheme) {}

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  QScheme qscheme() const override {
    return qscheme_;
  }

  double scale() const {
    return scale_;
  }

  int64_t zero_point() const {
    return zero_point_;
  }

  bool equalTo(QuantizerPtr other) override {
    if (!other.get() || other->qscheme() != qscheme_) {
      return false;
    }
    auto* other_per_tensor_affine =
        static_cast<PerTensorQuantizer*>(other.get());
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
  const QScheme qscheme_;
};

/**
 * PerChannelQuantizer is the same as PerTensorQuantizer
 * except that we have an independent scale and zero_point parameter
 * for each channel.
 *
 * Also note that per channel quantization is mostly applied to output channels
 * of weights since per-input channel of weight quantization or per-channel
 * quantization for activations can't be efficiently supported in most of
 * processors since it requires each multiplication result within a single
 * dot-product to have a different scale.
 */
struct TORCH_API PerChannelQuantizer : public UniformQuantizer {
  explicit PerChannelQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis,
      QScheme qscheme=kPerChannelAffine)
      : UniformQuantizer(scalar_type),
        scales_(scales),
        zero_points_(zero_points),
        axis_(axis),
        qscheme_(qscheme) {}

  QScheme qscheme() const override {
    return qscheme_;
  }

  Tensor scales() const {
    return scales_;
  }

  Tensor zero_points() const {
    return zero_points_;
  }

  int64_t axis() const {
    return axis_;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) override {
    if (!other.get() || other->qscheme() != qscheme_) {
      return false;
    }
    auto* other_per_channel_affine =
        static_cast<PerChannelQuantizer*>(other.get());
    return scalar_type() == other_per_channel_affine->scalar_type() &&
        scales().equal(other_per_channel_affine->scales()) &&
        zero_points().equal(other_per_channel_affine->zero_points()) &&
        axis() == other_per_channel_affine->axis();
  }

 protected:
  Tensor scales_;
  Tensor zero_points_;
  const int64_t axis_;
  const QScheme qscheme_;
};

/**
 * PerChannelFloatQParamsQuantizer is the same as PerChannelQuantizer
 * except that it expects both scale and zero point to be floating point values.
 *
 * This quantizer uses the kPerAffineChannelFloatQParams qscheme which is a variant of
 * kPerChannelAffine.
 *
 * The quantize equation in this case looks like -
 * Xq = (Xf - zero_point) * inv_scale, where inv_scale = 1.0/scale
 *
 * Note: Usage of floating point zero point is useful in cases where 0 doesn't need to
 * be exactly represented in the quantized space. We can get additional precision by
 * using floating point values for zero point.
 */
struct TORCH_API PerChannelFloatQParamsQuantizer : public PerChannelQuantizer {
  explicit PerChannelFloatQParamsQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : PerChannelQuantizer(scalar_type,
        scales,
        zero_points,
        axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffineFloatQParams;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) override {
    if (!other.get() || other->qscheme() != kPerChannelAffineFloatQParams) {
      return false;
    }
    auto* other_per_channel_float_qparams =
        static_cast<PerChannelFloatQParamsQuantizer*>(other.get());
    return scalar_type() == other_per_channel_float_qparams->scalar_type() &&
        scales().equal(other_per_channel_float_qparams->scales()) &&
        zero_points().equal(other_per_channel_float_qparams->zero_points()) &&
        axis() == other_per_channel_float_qparams->axis();
  }
};

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
TORCH_API QTensorImpl* get_qtensorimpl(const TensorBase& self);

// double and int64_t are because of the native function API, we only have these
// argument types right now in native functions
TORCH_API QuantizerPtr
make_per_tensor_affine_quantizer(
    double scale, int64_t zero_point, ScalarType scalar_type, QScheme qscheme=kPerTensorAffine);

TORCH_API QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type,
    QScheme qscheme=kPerChannelAffine);

// Create a Quantized Tensor given arguments for normal Tensor and a quantizer
TORCH_API Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

TORCH_API void set_quantizer_(const Tensor& self, ConstQuantizerPtr quantizer);

TORCH_API Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options);

TORCH_API Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options);

TORCH_API Tensor from_blob_quantized_per_channel_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const Tensor& scales,
    const Tensor& zero_points,
    const int64_t axis,
    const TensorOptions& options);

} // namespace at

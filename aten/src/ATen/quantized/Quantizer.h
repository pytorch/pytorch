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
#include <utility>

namespace at {

/**
 * UnknownQuantizer is a placeholder quantizer for functions that implement
 * quantization in a two step process.  First a tensor is allocated but with
 * unknown quantizer, and then the quantization kernel decides what the final
 * quantizer will be.
 */
struct TORCH_API UnknownQuantizer : public Quantizer {
  explicit UnknownQuantizer(ScalarType scalar_type)
    : Quantizer(scalar_type) {}

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;
  QScheme qscheme() const override;
  bool equalTo(QuantizerPtr other) const override;
};

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
 * AffineQuantizer uses affine transformation to do quantization.
 *
 * For quantize:
 * Y = clamp(round(X / scale + zero_point), min, max)
 * For dequantize:
 * X = (Y - zero_point) * scale
 */
struct TORCH_API AffineQuantizer : public UniformQuantizer {
  explicit AffineQuantizer(ScalarType scalar_type) : UniformQuantizer(scalar_type) {}
};

// Note that we will not have Symmetric Quantizer in backend to reduce
// complications in quantized kernel implementation.

/**
 * PerTensorAffineQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct TORCH_API PerTensorAffineQuantizer : public AffineQuantizer {
  explicit PerTensorAffineQuantizer(ScalarType scalar_type, double scale, int64_t zero_point)
    : AffineQuantizer(scalar_type),
        scale_(scale),
        zero_point_(zero_point) {}

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  QScheme qscheme() const override {
    return kPerTensorAffine;
  }

  double scale() const {
    return scale_;
  }

  int64_t zero_point() const {
    return zero_point_;
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerTensorAffine) {
      return false;
    }
    auto* other_per_tensor_affine =
        static_cast<PerTensorAffineQuantizer*>(other.get());
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
};

/**
 * PerChannelAffineQuantizer is the same as PerTensorAffineQuantizer
 * except that we have an independent scale and zero_point parameter
 * for each channel.
 *
 * Also note that per channel quantization is mostly applied to output channels
 * of weights since per-input channel of weight quantization or per-channel
 * quantization for activations can't be efficiently supported in most of
 * processors since it requires each multiplication result within a single
 * dot-product to have a different scale.
 */
struct TORCH_API PerChannelAffineQuantizer : public AffineQuantizer {
  explicit PerChannelAffineQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : AffineQuantizer(scalar_type),
        scales_(std::move(scales)),
        zero_points_(std::move(zero_points)),
        axis_(axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffine;
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

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffine) {
      return false;
    }
    auto* other_per_channel_affine =
        static_cast<PerChannelAffineQuantizer*>(other.get());
    return scalar_type() == other_per_channel_affine->scalar_type() &&
        scales().equal(other_per_channel_affine->scales()) &&
        zero_points().equal(other_per_channel_affine->zero_points()) &&
        axis() == other_per_channel_affine->axis();
  }

 protected:
  Tensor scales_;
  Tensor zero_points_;
  const int64_t axis_;
};

/**
 * PerChannelAffineFloatQParamsQuantizer is the same as PerChannelAffineQuantizer
 * except that it expects both scale and zero point to be floating point values.
 *
 * This quantizer uses the kPerChannelAffineFloatQParams qscheme which is a variant of
 * kPerChannelAffine.
 *
 * The quantize equation in this case looks like -
 * Xq = (Xf - zero_point) * inv_scale, where inv_scale = 1.0/scale
 *
 * Note: Usage of floating point zero point is useful in cases where 0 doesn't need to
 * be exactly represented in the quantized space. We can get additional precision by
 * using floating point values for zero point.
 */
struct TORCH_API PerChannelAffineFloatQParamsQuantizer : public PerChannelAffineQuantizer {
  explicit PerChannelAffineFloatQParamsQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : PerChannelAffineQuantizer(scalar_type,
        scales,
        zero_points,
        axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffineFloatQParams;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffineFloatQParams) {
      return false;
    }
    auto* other_per_channel_float_qparams =
        static_cast<PerChannelAffineFloatQParamsQuantizer*>(other.get());
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
    double scale, int64_t zero_point, ScalarType scalar_type);

TORCH_API QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type);

TORCH_API QuantizerPtr make_unknown_quantizer(ScalarType scalar_type);

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

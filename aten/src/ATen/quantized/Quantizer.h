#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <memory>

// TODO: move to c10 namespace after we
// unified caffe2::Tensor and at::Tensor

namespace at {

struct QTensorImpl;

using QTensor = Tensor;
using RealTensor = Tensor;

struct Quantizer;
using QuantizerPtr = c10::intrusive_ptr<Quantizer>;

/**
 * Quantizer is the class for storing all the information
 * that's necessary to perform quantize and dequantize
 * operation.
 *
 * We might have different types of quantization schemes and this is
 * the base class for all quantizers.
 *
 * QTensorImpl will hold a pointer to Quantizer so that we can support
 * different quantization schemes on Tensor.
 *
 * For example, the most common quantization scheme, Affine Quantization,
 * requires scale and zero_point as parameters, we'll store scale and zero_point
 * inside the instance and we can use it to quantize a float Tensor or
 * dequantize a quantized Tensor.
 *
 * When you add new types of leaf Quantizer class, please also
 * make sure to add a corresponding QScheme enum since
 * they should have one to one mapping.
 *
 * Note about intrusive_ptr:
 * QTensor holds an intrusive_ptr to Quantizer, and multiple Tensor can
 * share the same Quantizer. Quantizer should be immutable.
 */
struct CAFFE2_API Quantizer : public c10::intrusive_ptr_target {
  const QScheme qscheme_;
  explicit Quantizer(QScheme qscheme) : qscheme_(qscheme) {}
  virtual ~Quantizer();

  // Copied from torch/csrc/jit/scope.h
  QuantizerPtr intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Quantizer>::reclaim(this);
  }

  virtual QScheme qscheme() {
    return qscheme_;
  }

  /**
   * quantize a float Tensor into a quantized Tensor.
   */
  virtual QTensor quantize(RealTensor t) = 0;

  /**
   * dequantize a quantized Tensor into a float Tensor.
   */
  virtual RealTensor dequantize(QTensor t) = 0;
};

/**
 * UniformQuantizer is the parent class for all uniform quantizers.
 * These quantization scheme will map float value uniformly to
 * the quantized value. For example, affine quantizer is
 * the most commonly used scheme in this category.
 */
struct CAFFE2_API UniformQuantizer : public Quantizer {
  explicit UniformQuantizer(QScheme qscheme) : Quantizer(qscheme) {}
};

/**
 * NonUniformQuantizer is the parent class for all non-uniform quantizers.
 * These quantization scheme may map float value non-uniformly to the quantized
 * value. K-means quantization is a representative example in this category.
 */
struct CAFFE2_API NonUniformQuantizer : public Quantizer {
  explicit NonUniformQuantizer(QScheme qscheme) : Quantizer(qscheme) {}
};

// There is also StochasticQuantizer which is uniform but not affine

/**
 * AffineQuantizer uses affine transformation to do quantization.
 *
 * For quantize:
 * Y = clamp((X * scale + zero_point, min, max)
 * For dequantize:
 * X = (Y - zero_point) / scale
 */
struct CAFFE2_API AffineQuantizer : public UniformQuantizer {
  explicit AffineQuantizer(QScheme qscheme) : UniformQuantizer(qscheme) {}
};

/**
 * SymmetricQuantizer is similar to AffineQuantizer except that it
 * does not have zero_point
 *
 * For quantize:
 * Y = clamp(X * scale, min, max)
 * For dequantize:
 * X = Y / scale
 */
struct CAFFE2_API SymmetricQuantizer : public UniformQuantizer {
  explicit SymmetricQuantizer(QScheme qscheme) : UniformQuantizer(qscheme) {}
};

/**
 * PerTensorSymmetricQuantizer stores a single scale number which is
 * used for quantizing all the values in the given Tensor
 */
struct CAFFE2_API PerTensorSymmetricQuantizer : public SymmetricQuantizer {
  explicit PerTensorSymmetricQuantizer(float scale)
      : SymmetricQuantizer(kPerTensorSymmetric), scale_(scale) {}
  float scale_{1.0};
};

/**
 * PerChannelSymmetricQuantizer stores a vector of scale number and
 * applys symmetric quantization using different scales on each channel.
 *
 * Also note that per channel quantization is mostly applied to output channels
 * of weights since per-input channel of weight quantization or per-channel
 * quantization for activations can't be efficiently supported in most of
 * processors since it requires each multiplication result within a single
 * dot-product to have a different scale.
 */
struct CAFFE2_API PerChannelSymmetricQuantizer : public SymmetricQuantizer {
  explicit PerChannelSymmetricQuantizer(
      const std::vector<float>& scales,
      const std::vector<int64_t>& axis)
      : SymmetricQuantizer(kPerChannelSymmetric), scales_(scales), axis_(axis) {
    AT_CHECK(
        axis_.size() == 1,
        "Per channel symmetric quantization in multiple axis is not supported yet.");
  }

  std::vector<float> scales() const {
    return scales_;
  }

  std::vector<int64_t> axis() const {
    return axis_;
  }

 private:
  const std::vector<float> scales_;
  const std::vector<int64_t> axis_;
};

/**
 * PerTensorAffineQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct CAFFE2_API PerTensorAffineQuantizer : public AffineQuantizer {
  explicit PerTensorAffineQuantizer(float scale, uint8_t zero_point)
      : AffineQuantizer(kPerTensorAffine),
        scale_(scale),
        zero_point_(zero_point) {}

  QTensor quantize(RealTensor tensor) override;
  RealTensor dequantize(QTensor tensor) override;

  float scale() const {
    return scale_;
  }

  uint8_t zero_point() const {
    return zero_point_;
  }

 private:
  const float scale_;
  const uint8_t zero_point_;
};

/**
 * PerChannelAffineQuantizer is the same as PerTensorAffineQuantizer
 * except that we have an independent scale and zero_point parameter
 * for each channel.
 */
struct CAFFE2_API PerChannelAffineQuantizer : public AffineQuantizer {
  explicit PerChannelAffineQuantizer(
      const std::vector<float>& scales,
      const std::vector<uint8_t>& zero_points,
      const std::vector<int64_t>& axis)
      : AffineQuantizer(kPerChannelAffine),
        scales_(scales),
        zero_points_(zero_points),
        axis_(axis) {
    AT_CHECK(
        axis_.size() == 1,
        "Per channel affine quantization in multiple axis is not supported yet.");
  }

  std::vector<float> scales() const {
    return scales_;
  }

  std::vector<uint8_t> zero_points() const {
    return zero_points_;
  }

  std::vector<int64_t> axis() const {
    return axis_;
  }

 private:
  const std::vector<float> scales_;
  const std::vector<uint8_t> zero_points_;
  const std::vector<int64_t> axis_;
};

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
CAFFE2_API QTensorImpl* get_qtensorimpl(const QTensor& self);

// Quantize a float value into a uint8 value given scale and zero_point
CAFFE2_API qint8 quantize_uint8(float scale, uint8_t zero_point, float value);

// double and int64_t are because of the native function API, we only have these
// argument types right now in native functions
CAFFE2_API QuantizerPtr
make_per_tensor_affine_quantizer(double scale, int64_t zero_point);

// Create a QTensor given arguments for normal Tensor and a quantizer
CAFFE2_API QTensor new_qtensor_cpu(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

} // namespace at

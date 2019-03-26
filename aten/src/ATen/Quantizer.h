#pragma once

#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <ATen/core/Tensor.h>

#include <cmath>
#include <memory>

// TODO: move to c10 namespace after we
// unified caffe2::Tensor and at::Tensor

namespace at {

using QTensor = Tensor;
using RealTensor = Tensor;

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
 * For example, the most common quantization scheme, Affine Quantization, requires
 * scale and zero_point as parameters, we'll store scale and zero_point inside
 * the instance and we can use it to quantize a float Tensor or dequantize a
 * quantized Tensor.
 *
 * When you add new types of leaf Quantizer class, please also
 * make sure to add a corresponding QScheme enum since
 * they should have one to one mapping.
 */
struct C10_API Quantizer {
  c10::QScheme qscheme_;
  virtual ~Quantizer() {}

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
struct C10_API UniformQuantizer : public Quantizer {
  virtual ~UniformQuantizer() {}
};

/**
 * NonUniformQuantizer is the parent class for all non-uniform quantizers.
 * These quantization scheme may map float value non-uniformly to the quantized
 * value. K-means quantization is a representative example in this category.
 */
struct C10_API NonUniformQuantizer : public Quantizer {
  virtual ~NonUniformQuantizer() {}

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
struct C10_API AffineQuantizer : public UniformQuantizer {
  virtual ~AffineQuantizer() {}

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
struct C10_API SymmetricQuantizer : public UniformQuantizer {
  virtual ~SymmetricQuantizer() {}

};

/**
 * PerLayerSymmetricQuantizer stores a single scale number which is
 * used for quantizing all the values in the given Tensor
 */
struct C10_API PerLayerSymmetricQuantizer: public SymmetricQuantizer {
  PerLayerSymmetricQuantizer() {}
  PerLayerSymmetricQuantizer(float scale) : scale_(scale) {}
  virtual ~PerLayerSymmetricQuantizer() {}
  float scale_{1.0};
};

/**
 * PerChannelSymmetricQuantizer stores a vector of scale number and
 * applys symmetric quantization using different scales.
 */
struct C10_API PerChannelSymmetricQuantizer: public SymmetricQuantizer {
  PerChannelSymmetricQuantizer() {}
  virtual ~PerChannelSymmetricQuantizer() {}

  std::vector<float> scales_;
  int64_t channel_axis_;
};

/**
 * PerLayerAffineQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct C10_API PerLayerAffineQuantizer: public AffineQuantizer{
  PerLayerAffineQuantizer(float scale, uint8_t zero_point): scale_(scale), zero_point_(zero_point) {}

  virtual QTensor quantize(RealTensor tensor);
  virtual RealTensor dequantize(QTensor tensor);

  float scale() {
    return scale_;
  }

  int32_t zero_point() {
    return zero_point_;
  }

  float scale_{1.0};
  uint8_t zero_point_{0};
};

/**
 * PerChannelAffineQuantizer is the same as PerLayerAffineQuantizer
 * except that we have an independent scale and zero_point parameter
 * for each channel.
 */
struct C10_API PerChannelAffineQuantizer: public AffineQuantizer {
  PerChannelAffineQuantizer() {}

  std::vector<float> scales_;
  std::vector<uint8_t> zero_points_;
  int64_t channel_axis_;
};

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
QTensorImpl* get_qtensorimpl(const QTensor& self);

/* Some Helper Functions */
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}

qint8 QuantizeUint8(float scale, uint8_t zero_point, float value);

// double and int64_t are because of the native function API, we only have these argument
// types right now in native functions
std::shared_ptr<Quantizer> make_per_layer_affine_quantizer(double scale, int64_t zero_point);
QTensor new_qtensor(
    IntArrayRef sizes, const TensorOptions& options, float scale, int32_t zero_point);

} // namespace at

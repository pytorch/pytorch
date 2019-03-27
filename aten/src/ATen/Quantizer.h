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

struct QTensorImpl;

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
struct CAFFE2_API Quantizer {
  QScheme qscheme_;
  Quantizer() {}
  Quantizer(QScheme qscheme) : qscheme_(qscheme) {}
  virtual ~Quantizer();

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
  UniformQuantizer() {}
  UniformQuantizer(QScheme qscheme): Quantizer(qscheme) {}
  virtual ~UniformQuantizer();
};

/**
 * NonUniformQuantizer is the parent class for all non-uniform quantizers.
 * These quantization scheme may map float value non-uniformly to the quantized
 * value. K-means quantization is a representative example in this category.
 */
struct CAFFE2_API NonUniformQuantizer : public Quantizer {
  NonUniformQuantizer() {}
  NonUniformQuantizer(QScheme qscheme): Quantizer(qscheme) {}
  virtual ~NonUniformQuantizer();

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
  AffineQuantizer() {}
  AffineQuantizer(QScheme qscheme): UniformQuantizer(qscheme) {}
  virtual ~AffineQuantizer();
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
  SymmetricQuantizer() {}
  SymmetricQuantizer(QScheme qscheme): UniformQuantizer(qscheme) {}
  virtual ~SymmetricQuantizer();

};

/**
 * PerLayerSymmetricQuantizer stores a single scale number which is
 * used for quantizing all the values in the given Tensor
 */
struct CAFFE2_API PerLayerSymmetricQuantizer: public SymmetricQuantizer {
  PerLayerSymmetricQuantizer() {}
  PerLayerSymmetricQuantizer(float scale) : SymmetricQuantizer(kPerLayerSymmetric), scale_(scale) {}
  virtual ~PerLayerSymmetricQuantizer();
  float scale_{1.0};
};

/**
 * PerChannelSymmetricQuantizer stores a vector of scale number and
 * applys symmetric quantization using different scales.
 */
struct CAFFE2_API PerChannelSymmetricQuantizer: public SymmetricQuantizer {
  PerChannelSymmetricQuantizer() {}
  PerChannelSymmetricQuantizer(std::vector<float> scales, std::vector<int64_t> channel_axis): SymmetricQuantizer(kPerChannelSymmetric), scales_(scales), channel_axis_(channel_axis) {
    AT_ASSERT(channel_axis_.size() == 1);
  }
  virtual ~PerChannelSymmetricQuantizer();

  std::vector<float> scales_;
  std::vector<int64_t> channel_axis_;
};

/**
 * PerLayerAffineQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct CAFFE2_API PerLayerAffineQuantizer: public AffineQuantizer{
  PerLayerAffineQuantizer(float scale, uint8_t zero_point): AffineQuantizer(kPerLayerAffine), scale_(scale), zero_point_(zero_point) {}
  ~PerLayerAffineQuantizer();

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
struct CAFFE2_API PerChannelAffineQuantizer: public AffineQuantizer {
  PerChannelAffineQuantizer() {}
  PerChannelAffineQuantizer(std::vector<float> scales, std::vector<uint8_t> zero_points, std::vector<int64_t> channel_axis): AffineQuantizer(kPerChannelAffine), scales_(scales), zero_points_(zero_points), channel_axis_(channel_axis) {
    AT_ASSERT(channel_axis_.size() == 1);
  }
  ~PerChannelAffineQuantizer();

  std::vector<float> scales_;
  std::vector<uint8_t> zero_points_;
  std::vector<int64_t> channel_axis_;
};

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
CAFFE2_API QTensorImpl* get_qtensorimpl(const QTensor& self);

CAFFE2_API qint8 QuantizeUint8(float scale, uint8_t zero_point, float value);

// double and int64_t are because of the native function API, we only have these argument
// types right now in native functions
CAFFE2_API std::shared_ptr<Quantizer> make_per_layer_affine_quantizer(double scale, int64_t zero_point);
QTensor new_qtensor(
    IntArrayRef sizes, const TensorOptions& options, float scale, int8_t zero_point);

} // namespace at

#pragma once

#include <c10/core/QScheme.h>
#include <c10/core/MemoryFormat.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <cmath>
#include <memory>

// TODO: move to c10 namespace after we
// unified caffe2::Tensor and at::Tensor

namespace at {

class Tensor;
struct QTensorImpl;
struct Quantizer;
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;
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
 * Quantized Tensor holds an intrusive_ptr to Quantizer, and multiple Tensor can
 * share the same Quantizer. Quantizer should be immutable.
 */
struct CAFFE2_API Quantizer : public c10::intrusive_ptr_target {
  const QScheme qscheme_;
  const ScalarType scalar_type_;
  explicit Quantizer(QScheme qscheme, ScalarType scalar_type) : qscheme_(qscheme), scalar_type_(scalar_type) {}
  virtual ~Quantizer();

  // Copied from torch/csrc/jit/scope.h
  QuantizerPtr intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Quantizer>::reclaim(this);
  }

  QScheme qscheme() {
    return qscheme_;
  }

  ScalarType scalar_type() {
    return scalar_type_;
  }

  /**
   * quantize a float Tensor into a quantized Tensor.
   */
  virtual Tensor quantize(Tensor t) = 0;

  /**
   * dequantize a quantized Tensor into a float Tensor.
   */
  virtual Tensor dequantize(Tensor t) = 0;

  /**
   * Compare against `other` for equality.
   */
  virtual bool equalTo(QuantizerPtr other) = 0;
};

/**
 * UniformQuantizer is the parent class for all uniform quantizers.
 * These quantization scheme will map float value uniformly to
 * the quantized value. For example, affine quantizer is
 * the most commonly used scheme in this category.
 */
struct CAFFE2_API UniformQuantizer : public Quantizer {
  explicit UniformQuantizer(QScheme qscheme, ScalarType scalar_type) : Quantizer(qscheme, scalar_type) {}
};

/**
 * NonUniformQuantizer is the parent class for all non-uniform quantizers.
 * These quantization scheme may map float value non-uniformly to the quantized
 * value. K-means quantization is a representative example in this category.
 */
struct CAFFE2_API NonUniformQuantizer : public Quantizer {
  explicit NonUniformQuantizer(QScheme qscheme, ScalarType scalar_type) : Quantizer(qscheme, scalar_type) {}
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
struct CAFFE2_API AffineQuantizer : public UniformQuantizer {
  explicit AffineQuantizer(QScheme qscheme, ScalarType scalar_type) : UniformQuantizer(qscheme, scalar_type) {}
};

/**
 * SymmetricQuantizer is similar to AffineQuantizer except that it
 * does not have zero_point
 *
 * For quantize:
 * Y = clamp(round(X / scale), min, max)
 * For dequantize:
 * X = Y * scale
 */
struct CAFFE2_API SymmetricQuantizer : public UniformQuantizer {
  explicit SymmetricQuantizer(QScheme qscheme, ScalarType scalar_type) : UniformQuantizer(qscheme, scalar_type) {}
};

/**
 * PerTensorSymmetricQuantizer stores a single scale number which is
 * used for quantizing all the values in the given Tensor
 */
struct CAFFE2_API PerTensorSymmetricQuantizer : public SymmetricQuantizer {
  explicit PerTensorSymmetricQuantizer(ScalarType scalar_type, double scale)
    : SymmetricQuantizer(kPerTensorSymmetric, scalar_type), scale_(scale) {}
  double scale_{1.0};

  bool equalTo(QuantizerPtr other) override {
    if (auto* other_per_tensor_sym =
            dynamic_cast<PerTensorSymmetricQuantizer*>(other.get())) {
      return qscheme() == other_per_tensor_sym->qscheme() &&
          scalar_type() == other_per_tensor_sym->scalar_type() &&
          scale_ == other_per_tensor_sym->scale_;
    }
    return false;
  }
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
      ScalarType scalar_type,
      const std::vector<double>& scales,
      IntArrayRef axis)
    : SymmetricQuantizer(kPerChannelSymmetric, scalar_type), scales_(scales), axis_(axis.vec()) {
    TORCH_CHECK(
        axis_.size() == 1,
        "Per channel symmetric quantization in multiple axis is not supported yet.");
  }

  std::vector<double> scales() const {
    return scales_;
  }

  IntArrayRef axis() const {
    return axis_;
  }

  bool equalTo(QuantizerPtr other) override {
    if (auto* other_per_channel_sym =
            dynamic_cast<PerChannelSymmetricQuantizer*>(other.get())) {
      return qscheme() == other_per_channel_sym->qscheme() &&
          scalar_type() == other_per_channel_sym->scalar_type() &&
          scales() == other_per_channel_sym->scales() &&
          axis() == other_per_channel_sym->axis();
    }
    return false;
  }

 private:
  const std::vector<double> scales_;
  const SmallVector<int64_t, 1> axis_;
};

/**
 * PerTensorAffineQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct CAFFE2_API PerTensorAffineQuantizer : public AffineQuantizer {
  explicit PerTensorAffineQuantizer(ScalarType scalar_type, double scale, int64_t zero_point)
    : AffineQuantizer(kPerTensorAffine, scalar_type),
        scale_(scale),
        zero_point_(zero_point) {}

  Tensor quantize(Tensor tensor) override;
  Tensor dequantize(Tensor tensor) override;

  double scale() const {
    return scale_;
  }

  int64_t zero_point() const {
    return zero_point_;
  }

  bool equalTo(QuantizerPtr other) override {
    if (auto* other_per_tensor_affine =
            dynamic_cast<PerTensorAffineQuantizer*>(other.get())) {
      return qscheme() == other_per_tensor_affine->qscheme() &&
          scalar_type() == other_per_tensor_affine->scalar_type() &&
          scale() == other_per_tensor_affine->scale() &&
          zero_point() == other_per_tensor_affine->zero_point();
    }
    return false;
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
 */
struct CAFFE2_API PerChannelAffineQuantizer : public AffineQuantizer {
  explicit PerChannelAffineQuantizer(
      ScalarType scalar_type,
      const std::vector<double>& scales,
      const std::vector<int64_t>& zero_points,
      IntArrayRef axis)
    : AffineQuantizer(kPerChannelAffine, scalar_type),
    scales_(scales),
    zero_points_(zero_points),
    axis_(axis.vec()) {
    TORCH_CHECK(
        axis_.size() == 1,
        "Per channel affine quantization in multiple axis is not supported yet.");
  }

  std::vector<double> scales() const {
    return scales_;
  }

  std::vector<int64_t> zero_points() const {
    return zero_points_;
  }

  IntArrayRef axis() const {
    return axis_;
  }

  Tensor quantize(Tensor tensor) override;
  Tensor dequantize(Tensor tensor) override;

  bool equalTo(QuantizerPtr other) override {
    if (auto* other_per_channel_affine =
            dynamic_cast<PerChannelAffineQuantizer*>(other.get())) {
      return qscheme() == other_per_channel_affine->qscheme() &&
          scalar_type() == other_per_channel_affine->scalar_type() &&
          scales() == other_per_channel_affine->scales() &&
          zero_points() == other_per_channel_affine->zero_points() &&
          axis() == other_per_channel_affine->axis();
    }
    return false;
  }

 private:
  const std::vector<double> scales_;
  const std::vector<int64_t> zero_points_;
  const SmallVector<int64_t, 1> axis_;
};

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
CAFFE2_API QTensorImpl* get_qtensorimpl(const Tensor& self);

// Quantize a float value into a uint value given scale and zero_point
template <typename T>
CAFFE2_API T quantize_val(double scale, int64_t zero_point, float value);
template <typename T>
CAFFE2_API Tensor quantize_tensor(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template <typename T>
CAFFE2_API float dequantize_val(double scale, int64_t zero_point, T value);
template <typename T>
CAFFE2_API Tensor dequantize_tensor(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point);
template <typename SRC_T, typename DST_T>
CAFFE2_API DST_T requantize_val(double, int64_t, double, int64_t, SRC_T src);

// double and int64_t are because of the native function API, we only have these
// argument types right now in native functions
CAFFE2_API QuantizerPtr
make_per_tensor_affine_quantizer(
    double scale, int64_t zero_point, ScalarType scalar_type);

CAFFE2_API QuantizerPtr
make_per_channel_affine_quantizer(
    const std::vector<double>& scales, const std::vector<int64_t>& zero_points,
    IntArrayRef axis, ScalarType scalar_type);

// Create a Quantized Tensor given arguments for normal Tensor and a quantizer
CAFFE2_API Tensor new_qtensor_cpu(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer,
    MemoryFormat memory_format);

} // namespace at

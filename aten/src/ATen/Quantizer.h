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

// Quantizer is an object that stores all the information
// that's necessary to perform quantize and dequantize
// operation
struct C10_API Quantizer {
  // QuantizerTypeId type_id_;
  /* used for rounding */
  int32_t num_bits_;

  virtual ~Quantizer() {}

  virtual std::string name() {
    return "Quantizer";
  }

  virtual QTensor quantize(RealTensor t) = 0;

  virtual RealTensor dequantize(QTensor t) = 0;
};

struct C10_API UniformQuantizer : public Quantizer {
  virtual ~UniformQuantizer() {}
  virtual std::string name() {
    return "UniformQuantizer";
  }
};

struct C10_API NonUniformQuantizer : public Quantizer {
  virtual ~NonUniformQuantizer() {}

  virtual std::string name() {
    return "NonUniformQuantizer";
  }
};

// There is also StochasticQuantizer which is uniform but not affine

struct C10_API AffineQuantizer : public UniformQuantizer {
  virtual ~AffineQuantizer() {}

  virtual std::string name() {
    return "AffineQuantizer";
  }
};

struct C10_API SymmetricQuantizer : public UniformQuantizer {
  virtual ~SymmetricQuantizer() {}

  virtual std::string name() {
    return "SymmetricQuantizer";
  }
};

struct C10_API PerLayerSymmetricQuantizer: public SymmetricQuantizer {
  PerLayerSymmetricQuantizer() {}
  PerLayerSymmetricQuantizer(float scale) : scale_(scale) {}
  virtual ~PerLayerSymmetricQuantizer() {}
  virtual std::string name() {
    return "PerLayerSymmetricQuantizer";
  }
  float scale_{1.0};
};

struct C10_API PerChannelSymmetricQuantizer: public SymmetricQuantizer {
  PerChannelSymmetricQuantizer() {}
  virtual ~PerChannelSymmetricQuantizer() {}

  virtual std::string name() {
    return "PerChannelSymmetricQuantizer";
  }

  std::vector<float> scales_;
  std::vector<int32_t> zero_points_;
};

template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}

qint8 QuantizeUint8(float scale, int32_t zero_point, float value);

void ChooseParams(RealTensor tensor, float* r_scale, int* r_zero_point);

struct C10_API PerLayerAffineQuantizer: public AffineQuantizer{
  PerLayerAffineQuantizer(float scale, int32_t zero_point): scale_(scale), zero_point_(zero_point) {}

  virtual std::string name() {
    return "PerLayerAffineQuantizer";
  }

  virtual QTensor quantize(RealTensor tensor);
  virtual RealTensor dequantize(QTensor tensor);

  float scale() {
    return scale_;
  }

  int32_t zero_point() {
    return zero_point_;
  }

  float scale_{1.0};
  int32_t zero_point_{0};
};

struct C10_API PerChannelAffineQuantizer: public AffineQuantizer {
  PerChannelAffineQuantizer() {}

  virtual std::string name() {
    return "PerChannelAffineQuantizer";
  }

  std::vector<float> scales_;
  std::vector<int32_t> zero_points_;
};

std::shared_ptr<Quantizer> make_per_layer_affine_quantizer(double scale, int64_t zero_point);
QTensor new_qtensor(
    IntList sizes, const TensorOptions& options, float scale, int32_t zero_point);
//std::unique_ptr<Quantizer> create_quantizer(QScheme qscheme);

} // namespace at

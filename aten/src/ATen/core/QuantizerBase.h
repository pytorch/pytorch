#pragma once

#include <c10/core/ScalarType.h>
#include <c10/core/QScheme.h>
#include <c10/util/intrusive_ptr.h>

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
struct TORCH_API Quantizer : public c10::intrusive_ptr_target {
  const ScalarType scalar_type_;
  explicit Quantizer(ScalarType scalar_type) : scalar_type_(scalar_type) {}
  virtual ~Quantizer();

  // Copied from torch/csrc/jit/ir/scope.h
  QuantizerPtr intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Quantizer>::reclaim(this);
  }

  /**
   * Each concrete Quantizer type should have a unique QScheme type.
   */
  virtual QScheme qscheme() const = 0;

  ScalarType scalar_type() {
    return scalar_type_;
  }

  /**
   * quantize a float Tensor into a quantized Tensor.
   */
  virtual Tensor quantize(const Tensor& t) = 0;

  /**
   * dequantize a quantized Tensor into a float Tensor.
   */
  virtual Tensor dequantize(const Tensor& t) = 0;

  /**
   * dequantize a quantized Tensor into a float Tensor, out= variant
   */
  virtual Tensor& dequantize_out(Tensor& out, const Tensor& t) = 0;

  /**
   * Compare against `other` for equality.
   */
  virtual bool equalTo(QuantizerPtr other) = 0;
};

} // namespace at

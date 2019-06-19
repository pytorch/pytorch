#pragma once

#include <ATen/core/ATenGeneral.h>
#include <c10/core/Allocator.h>
#include <c10/util/Deprecated.h>
#include <ATen/core/Generator.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h>
#include <c10/core/TensorTypeIdRegistration.h>
#include <ATen/core/Reduction.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>

#include <c10/util/Optional.h>

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>

// To solve the conflict of s_addr in inaddr.h
#ifdef _MSC_VER
#ifdef s_addr
#undef s_addr
#endif
#endif

namespace c10 {
struct Storage;
}

namespace at {

class Tensor;
using TensorList = ArrayRef<Tensor>;

class Context;
struct Generator;

struct Quantizer;
// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;

static inline void noop_deleter(void*) {}

enum class TypeID {
  ${type_ids}
  ComplexCPU,
  Undefined,
  NumOptions
};

struct CAFFE2_API Type {
  explicit Type() {}

  virtual ~Type() {}
  virtual const char * toString() const = 0;
  virtual TypeID ID() const = 0;
};

} // namespace at

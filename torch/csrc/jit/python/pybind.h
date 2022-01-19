#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace torch {
namespace jit {

// This is a variant of shared_ptr that "sees through" a wrapper.
// We use it to convert Value, Node, Block and node to "wrapped" Python
// values. When we destruct the C++ object, the wrapper's pointer will
// be set to 0 and any future dereferencing will throw. We need this
// because the Python objects may hang around after the C++ object
// has already been destroyed.
// This also needs the magic type_caster below, which is from the
// workaround offered in https://github.com/pybind/pybind11/issues/2751
template <typename T>
class unwrapping_shared_ptr {
  static_assert(
      std::is_same<T, torch::jit::Value>::value ||
          std::is_same<T, torch::jit::Node>::value ||
          std::is_same<T, torch::jit::Block>::value,
      "unwrapping type only defined for Graph object types");

 private:
  std::shared_ptr<torch::jit::Wrap<T>> impl;

 public:
  unwrapping_shared_ptr() : impl({}) {}
  explicit unwrapping_shared_ptr(T* p) : impl(p->wrap()) {
    impl->clear_cb = &clear_registered_instances;
  }
  T* get() const {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return impl->elem;
  }
  // we need to disable the overloaded & for PyBind11 < 2.3 due.
  // see https://github.com/pybind/pybind11/pull/1435
#if (PYBIND11_VERSION_MAJOR > 2) || \
    ((PYBIND11_VERSION_MAJOR == 2) && (PYBIND11_VERSION_MINOR >= 3))
  T** operator&() {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return &(impl->elem);
  }
#endif
};

} // namespace jit
} // namespace torch

PYBIND11_DECLARE_HOLDER_TYPE(T, torch::jit::unwrapping_shared_ptr<T>, true);

namespace pybind11 {
namespace detail {

#define CREATE_UNWRAPPING_CASTER(Class)                                                   \
  template <>                                                                             \
  struct type_caster<Class> : public type_caster_base<Class> {                            \
   public:                                                                                \
    using type = Class;                                                                   \
    using holder_type = torch::jit::unwrapping_shared_ptr<Class>;                         \
                                                                                          \
    bool load(handle src, bool convert) {                                                 \
      return load_impl<type_caster<Class>>(src, convert);                                 \
    }                                                                                     \
                                                                                          \
    explicit operator type*() {                                                           \
      return static_cast<type*>(value);                                                   \
    }                                                                                     \
    explicit operator type&() {                                                           \
      return *static_cast<type*>(value);                                                  \
    }                                                                                     \
                                                                                          \
   protected:                                                                             \
    friend class type_caster_generic;                                                     \
                                                                                          \
    bool load_value(value_and_holder&& v_h) {                                             \
      if (v_h.holder_constructed()) {                                                     \
        value = v_h.template holder<holder_type>().get();                                 \
        return true;                                                                      \
      } else {                                                                            \
        throw cast_error(                                                                 \
            "Unable to cast from non-held to held instance (#Class& to Holder<#Class>)"); \
      }                                                                                   \
    }                                                                                     \
  }

CREATE_UNWRAPPING_CASTER(torch::jit::Node);
CREATE_UNWRAPPING_CASTER(torch::jit::Value);
CREATE_UNWRAPPING_CASTER(torch::jit::Block);

#undef CREATE_UNWRAPPING_CASTER

} // namespace detail
} // namespace pybind11

namespace pybind11 {
namespace detail {

template <>
struct type_caster<torch::jit::IValue> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::IValue, _("IValue"));

  bool load(handle src, bool) {
    try {
      value = torch::jit::toTypeInferredIValue(src);
      return true;
    } catch (std::exception& e) {
      return false;
    }
  }

  static handle cast(
      torch::jit::IValue src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return torch::jit::toPyObject(std::move(src)).release();
  }
};

template <>
struct type_caster<torch::jit::Symbol> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::Symbol, _("Symbol"));

  bool load(handle src, bool) {
    // TODO: Is there a way to py::cast that doesn't raise an exception on
    // failure?  Can we catch pybind11::cast_error here instead?
    std::string src_str;
    try {
      src_str = py::cast<std::string>(src);
    } catch (std::exception& e) {
      return false;
    }
    value = torch::jit::Symbol::fromQualString(src_str);
    return true;
  }

  static handle cast(
      torch::jit::Symbol src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(std::string(src.toQualString()), return_value_policy::copy)
        .release();
  }
};

template <>
struct type_caster<torch::jit::AttributeKind> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::AttributeKind, _("AttributeKind"));

  bool load(handle src, bool) {
    return false;
  }

  static handle cast(
      torch::jit::AttributeKind src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(
               std::string(torch::jit::toString(src)),
               return_value_policy::copy)
        .release();
  }
};

// See https://github.com/pybind/pybind11/issues/637
using ListCasterBase = pybind11::detail::
    list_caster<std::vector<torch::jit::Node*>, torch::jit::Node*>;
template <>
struct type_caster<std::vector<torch::jit::Node*>> : ListCasterBase {
  static handle cast(
      const std::vector<torch::jit::Node*>& src,
      return_value_policy,
      handle parent) {
    return ListCasterBase::cast(src, return_value_policy::reference, parent);
  }
  static handle cast(
      const std::vector<torch::jit::Node*>* src,
      return_value_policy pol,
      handle parent) {
    return cast(*src, pol, parent);
  }
};

} // namespace detail
} // namespace pybind11

namespace torch {
namespace jit {

static inline py::tuple tuple_tail(const py::tuple& tup) {
  py::tuple r(tup.size() - 1);
  for (const auto i : c10::irange(1, tup.size())) {
    r[i - 1] = tup[i];
  }
  return r;
}

} // namespace jit
} // namespace torch

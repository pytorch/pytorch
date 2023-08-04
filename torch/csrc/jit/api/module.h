#pragma once
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/named_value.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/api/include/torch/ordered_dict.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/utils/memory.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// This file contains classes which assist in desugaring Python style
// modules and their methods into flattened graphs which don't have any
// function calls.

namespace torch {
namespace jit {

using ::c10::Argument;
using ::c10::FunctionSchema;
using ::c10::QualifiedName;
// Map which stores filename to content.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

using ModulePtr = c10::intrusive_ptr<c10::ivalue::Object>;

struct Module;

template <typename T>
struct slot_list_impl;

template <typename T>
struct Named {
  std::string name;
  T value;
};

using NameModule = Named<Module>;
using NameValue = Named<IValue>;
using NameTensor = Named<at::Tensor>;

namespace detail {
struct TORCH_API ModulePolicy;
struct TORCH_API ParameterPolicy;
struct TORCH_API AttributePolicy;
struct TORCH_API BufferPolicy;
template <typename P>
struct NamedPolicy;
} // namespace detail

using module_list = slot_list_impl<detail::ModulePolicy>;
using named_module_list =
    slot_list_impl<detail::NamedPolicy<detail::ModulePolicy>>;

using parameter_list = slot_list_impl<detail::ParameterPolicy>;
using named_parameter_list =
    slot_list_impl<detail::NamedPolicy<detail::ParameterPolicy>>;

using attribute_list = slot_list_impl<detail::AttributePolicy>;
using named_attribute_list =
    slot_list_impl<detail::NamedPolicy<detail::AttributePolicy>>;

using buffer_list = slot_list_impl<detail::BufferPolicy>;
using named_buffer_list =
    slot_list_impl<detail::NamedPolicy<detail::BufferPolicy>>;

using ModuleLookup = std::function<Module(const std::vector<std::string>&)>;

struct TORCH_API Module : public Object {
  explicit Module(c10::QualifiedName class_name);
  Module(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);
  Module() = default;
  Module(const Module&) = default;
  Module& operator=(const Module&) = default;
  Module(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);
  Module(ModulePtr module_value) : Object(std::move(module_value)) {}
  ~Module() = default;

  void set_optimized(bool o) {
    TORCH_WARN(
        "Module::set_optimized() is deprecated and has no effect. "
        "Please use setGraphExecutorOptimize()");
  }

  bool is_optimized() const {
    TORCH_WARN(
        "Module::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  IValue forward(std::vector<IValue> inputs, const Kwargs& kwargs = Kwargs()) {
    return get_method("forward")(std::move(inputs), kwargs);
  }

  // In script modules, buffers are Tensors attribute that are _not_ registered
  // as parameters. This is different than in nn.Module where there is a special
  // register_buffer method. With this simplification, we only need to track
  // whether a slot is a parameter to be able to classify it.
  void register_buffer(const std::string& name, at::Tensor v) {
    bool is_param = false;
    bool is_buffer = true;
    type()->addOrCheckAttribute(name, TensorType::get(), is_param, is_buffer);
    _ivalue()->setAttr(name, std::move(v));
  }

  void register_parameter(
      const std::string& name,
      at::Tensor v,
      bool is_buffer) {
    type()->addOrCheckAttribute(name, TensorType::get(), !is_buffer, is_buffer);
    _ivalue()->setAttr(name, std::move(v));
  }

  void register_attribute(
      const std::string& name,
      const TypePtr& t,
      IValue v,
      bool is_param = false,
      bool is_buffer = false) {
    type()->addOrCheckAttribute(name, t, is_param, is_buffer);
    _ivalue()->setAttr(name, std::move(v));
  }

  void register_module(const std::string& name, const Module& module) {
    type()->addOrCheckAttribute(name, module.type());
    _ivalue()->setAttr(name, module._ivalue());
  }

  void apply(const std::function<void(Module&)>& fn);

  buffer_list buffers(bool recurse = true) const;
  named_buffer_list named_buffers(bool recurse = true) const;

  module_list children() const; // direct modules
  named_module_list named_children() const;
  module_list modules() const; // all modules, including this one, recursively
  named_module_list named_modules() const;

  // all tensors involved in gradient optimization
  parameter_list parameters(bool recurse = true) const;
  named_parameter_list named_parameters(bool recurse = true) const;

  // all members of the object, similar to iterating over dir(obj) in python
  attribute_list attributes(bool recurse = true) const;
  named_attribute_list named_attributes(bool recurse = true) const;

  void dump(
      bool print_method_bodies,
      bool print_attr_values,
      bool print_param_values) const;

  std::string dump_to_str(
      bool print_method_bodies,
      bool print_attr_values,
      bool print_param_values) const;

  /// Enables "training" mode.
  void train(bool on = true);
  /// Calls train(false) to enable "eval" mode.
  /// Do not override this method, override `train()` instead.
  void eval() {
    train(/*on=*/false);
  }
  /// True if the module is in training mode.
  bool is_training() const {
    return attr("training", true).toBool();
  }

  /// Recursively casts all parameters to the given `dtype` and `device`.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  void to(at::Device device, at::ScalarType dtype, bool non_blocking = false);

  /// Recursively casts all parameters to the given dtype.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  void to(at::ScalarType dtype, bool non_blocking = false);

  /// Recursively moves all parameters to the given device.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  void to(at::Device device, bool non_blocking = false);

  void save(
      std::ostream& out,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  void save(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  void _save_for_mobile(
      std::ostream& out,
      const ExtraFilesMap& extra_files = ExtraFilesMap(),
      bool save_mobile_debug_info = false,
      bool use_flatbuffer = false) const;

  void _save_for_mobile(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap(),
      bool save_mobile_debug_info = false,
      bool use_flatbuffer = false) const;

  Module copy() const;

  Module deepcopy() const;

  // Clones both the underlying `ClassType` and the module instance(data), this
  // function creates a new `ClassType` and returns a new instance that has the
  // same data as the current instance but with the new type, shared ClassType
  // will be preserved as well
  Module clone(bool inplace = false) const;

  // Clones both the underlying `ClassType` and the module instance(data), this
  // function creates a new `ClassType` and returns a new instance that has the
  // same data as the current instance but with the new type, shared ClassType
  // will be preserved as well. Also allows the caller to specify a set of
  // method and attribute names to not clone.
  Module clone(
      bool inplace,
      const std::unordered_set<std::string>& ignored_method,
      const std::unordered_set<std::string>& ignored_attributes) const;

  void clone_method(const Module& orig, const std::string& name);

  IValue operator()(std::vector<IValue> inputs);

  template <typename... Types>
  IValue create_class(const c10::QualifiedName& name, Types&&... args) const {
    return create_class(name, {IValue(std::forward<Types>(args))...});
  }

  IValue create_class(const c10::QualifiedName& name, Stack stack) const;

  inline bool operator==(const Module& y) const noexcept {
    return _ivalue() == y._ivalue();
  }

  void set_delete_memory(std::shared_ptr<char> delete_mem) {
    mem_to_delete_ = std::move(delete_mem);
  }

  // A set of functions to maintain input shapes through torch.jit.save and
  // torch.jit.load. It only works on tensors and lists/dicts of tensors
  // because tracing is only supported by these types.
  void store_traced_inputs(std::string func_name, std::vector<IValue> inputs) {
    if (inputs.size() == 0) {
      return;
    }
    auto c10_inputs = c10::impl::GenericList(AnyType::get());
    for (IValue& value : inputs) {
      // Not checking whether this is traceable type as that is already checked
      // higher up in the stack and changing that would require a larger
      // restructuring.
      c10_inputs.emplace_back(std::move(value));
    }
    traced_inputs_.insert_or_assign(func_name, c10_inputs);
  }

  c10::Dict<std::string, c10::impl::GenericList> retrieve_traced_inputs()
      const {
    return traced_inputs_;
  }

 private:
  Module clone_impl(
      std::unordered_map<TypePtr, TypePtr>& type_remap,
      bool inplace,
      IValue::HashAliasedIValueMap memo,
      const std::unordered_set<std::string>& ignored_methods,
      const std::unordered_set<std::string>& ignored_attributes) const;

  void clone_method(
      const Module& orig,
      const Function& method,
      const std::unordered_map<TypePtr, TypePtr>& type_remap);

  c10::QualifiedName getNameForMethod(std::string basename) const {
    return QualifiedName(*type()->name(), std::move(basename));
  }

  void to_impl(
      const c10::optional<at::Device>& device,
      const c10::optional<at::ScalarType>& dtype,
      bool non_blocking);

  // Extra handle for the module to delete when itself is deleted
  std::shared_ptr<char> mem_to_delete_;

  // Map of function names to the traced inputs that they have been traced with
  c10::Dict<std::string, c10::impl::GenericList> traced_inputs_;
};

// C++ equivalent api of `torch.jit.freeze`. See documentation there for
// details.
TORCH_API Module freeze(
    const Module& module,
    const c10::optional<std::vector<std::string>>& preserved_attrs =
        c10::nullopt,
    bool optimize_numerics = true);

// C++ equivalent api of `torch.jit.optimize_for_inference`. See documentation
// there for details.
TORCH_API Module optimize_for_inference(
    Module& module,
    const std::vector<std::string>& other_methods = {});

enum class FusionBehavior { STATIC, DYNAMIC };

using FusionStrategy = std::vector<std::pair<FusionBehavior, size_t>>;
// clang-format off
/*
Sets the type and number of specializations that can occur during fusion.

Usage: provide a list of pairs (type, depth) where type is one of STATIC or DYNAMIC
and depth is an integer.

Behavior - static vs dynamic:
    In STATIC fusion, fused ops are compiled to have fixed input shapes. The shape is determined
    based on some initial profiling runs.
    In DYNAMIC fusion, fused ops are compiled to have variable input shapes, so that multiple
    shapes are possible.

In both cases, we also recompile on new striding behavior, device, or dtype.

Behavior - fallback functions & depth:
    When an input doesn't match the format required by the specialized compiled op, it will run
    a fallback function. Fallback functions are recursively be compiled and specialized based
    on the observed tensor shapes. Since compilation can be slow, the "depth" parameter is provided to
    limit the number of specializations that can be compiled, before giving up on recompiling and
    falling back to a completely un-fused, un-specialized implementation.

The list of (type, depth) pairs controls the type of specializations and the number of
specializations. For example: [(STATIC, 2), (DYNAMIC, 2)] indicates that the first
two specializations will use static fusions, the following two specializations will use
dynamic fusion, and any inputs that satisfy none of the 4 options will run an
unfused implementation.

NB: in the future, if more as more fusion backends are added there may be more granular
apis for specific fusers.
*/
// clang-format on
TORCH_API FusionStrategy getFusionStrategy();
// returns previous strategy
TORCH_API FusionStrategy setFusionStrategy(FusionStrategy& fusion_strategy);

namespace detail {

struct TORCH_API SlotCursor {
  Module module_;
  int64_t i_; // slot offset, -1 indicates the module itself
};

} // namespace detail

// This iterator allows the (optionally recursive) enumeration of
// the  members of a Module. It performs a depth-first pre-order
// traversal of the module. The Policy template parameter determines
// which slots of the object should be included. For instance,
// when iterating parameters, we return the parameter tensors,
// but skip modules, buffers, and other attributes.
// See ModulePolicy for comments about Policy object's API.
template <typename Policy>
struct slot_iterator_impl {
  using SlotCursor = detail::SlotCursor;
  using value_type = typename Policy::value_type;
  slot_iterator_impl(
      Module root,
      bool recurse, // if true, do a depth-first search, otherwise, just look at
                    // slots of root
      bool return_module) // if true include root itself as the first thing
                          // visited (used in modules())
      : cursors_({SlotCursor{std::move(root), return_module ? -1 : 0}}),
        recurse_(recurse) {
    // advance iterator to first valid element (or the end, if empty)
    while_not_valid_next();
  }
  // empty cursors_, represents end of iteration
  slot_iterator_impl() : recurse_(false) {}
  value_type operator*() const {
    return Policy::create(cursors_, cur());
  }
  value_type operator->() const {
    return **this;
  }
  slot_iterator_impl& operator++() {
    next_valid();
    return *this;
  }
  slot_iterator_impl operator++(int) {
    // this is really expensive, should we delete it so people don't use it
    // instead of prefix?
    slot_iterator_impl old = *this;
    ++(*this);
    return old;
  }

 private:
  // return_module() is a corner case where instead of returning a submodule
  // of root, we are returning root itself, because we are iterating modules(),
  // which contains the root module itself.
  // It is represented with a single SlotCursor whose index is -1.
  bool return_module() const {
    return top().i_ == -1;
  }
  const SlotCursor& top() const {
    return cursors_.back();
  }
  SlotCursor& top() {
    return cursors_.back();
  }
  IValue cur() const {
    return return_module() ? top().module_._ivalue()
                           : top().module_._ivalue()->getSlot(top().i_);
  }

  // advance to the next slot in a depth first pre-order traversal of the
  // modules slots. This function does not guarantee the next slot is a
  // valid element of the iteration. That is done by valid().
  // invariant: !cursors_.empty()
  void next() {
    // we just returned the module itself, advance i_ to 0 so we are now
    // at the first slot of the module.
    if (return_module()) {
      ++top().i_;
      return;
    }
    // the last traversal action advanced beyond the number of slots in the
    // module so continue the iteration in the parent.
    if (top().i_ >= int64_t(top().module_._ivalue()->type()->numAttributes())) {
      cursors_.pop_back();
      if (!cursors_.empty()) {
        ++top().i_;
      }
      return;
    }
    // if the current thing is a module, we have to scan it for recursive
    // traversals. We do this by adding a new SlotCursor to track the traversal.
    if (recurse_ &&
        top().module_._ivalue()->type()->getAttribute(top().i_)->is_module()) {
      cursors_.emplace_back(SlotCursor{cur().toModule(), 0});
      return;
    }
    // common case: advance to the next slot.
    ++top().i_;
  }
  // is the current position of the iterator a valid one?
  // otherwise, we have to continue advancing.
  bool valid() const {
    return top().i_ <
        int64_t(top().module_._ivalue()->type()->numAttributes()) &&
        Policy::valid(
               top().module_._ivalue()->type(),
               top().i_,
               top().module_._ivalue()->getSlot(top().i_));
  }
  void while_not_valid_next() {
    // advance iteration until we are either at the end (cursors_.empty())
    // or in a valid state. return_module() is a special case,
    // and is always considered valid, regardless of Policy, because it is
    // it is only true when we are iterating modules.
    while (!cursors_.empty() && !return_module() && !valid()) {
      next();
    }
  }
  void next_valid() {
    // avoid crashing if this is empty
    if (cursors_.empty()) {
      return;
    }
    // advance to next element, which is maybe not valid
    next();
    while_not_valid_next();
  }

  std::vector<SlotCursor> cursors_;
  bool recurse_;

  friend inline bool operator!=(
      const slot_iterator_impl<Policy>& a,
      const slot_iterator_impl<Policy>& b) {
    // we are finished iteration when we have no more iteration SlotCursors.
    // end is always an empty iterator with no cursors.
    return (a.cursors_.empty() != b.cursors_.empty());
  }
};

// This type represents lists of parameters, attributes, and
// submodules contained in the module. It is abstract because
// they are not stored directly in std::vectors but inside the
// module's IValue object itself.
template <typename Policy>
struct slot_list_impl {
  using iterator = slot_iterator_impl<Policy>;
  using const_iterator = slot_iterator_impl<Policy>;
  using value_type = typename iterator::value_type;
  slot_iterator_impl<Policy> begin() const {
    return slot_iterator_impl<Policy>(module_, recurse_, return_module_);
  }
  slot_iterator_impl<Policy> end() const {
    return slot_iterator_impl<Policy>();
  }
  size_t size() const {
    if (!size_) {
      size_ = size_t(0);
      // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
      for (const value_type& s : *(this)) {
        (void)s; // Suppress unused variable warning
        ++*size_;
      }
    }
    return *size_;
  }

  slot_list_impl(Module module, bool recurse, bool return_module)
      : module_(std::move(module)),
        recurse_(recurse),
        return_module_(return_module),
        size_(c10::nullopt) {
    if (!recurse && !return_module && Policy::all_slots) {
      size_ = module_.num_slots();
    }
  }

 private:
  Module module_;
  bool recurse_;
  bool return_module_;
  // size of this list, cached on first request
  // when we need to filter the slot list
  mutable c10::optional<size_t> size_;
  friend struct Module;
};

namespace detail {

// slot_iterator_impl always iterate over all the slots in a module,
// the Policy template argument determines slots should be returned and their
// types
struct TORCH_API ModulePolicy {
  // the type of the value being returned
  using value_type = Module;

  // the logic for creating the type being returned, given the raw IValue
  // of that object.
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return Module(std::move(v).toObject());
  }
  // is slot i in typ something that this iterator should return, otherwise,
  // we skip it.
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return typ->getAttribute(i)->is_module();
  }
  // are we going to return everything? If so, we can optimize the calculate
  // of the size of the list.
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

struct TORCH_API ParameterPolicy {
  using value_type = at::Tensor;
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return std::move(v).toTensor();
  }
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return typ->is_parameter(i) && v.isTensor();
  }
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

struct TORCH_API BufferPolicy {
  using value_type = at::Tensor;
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return std::move(v).toTensor();
  }
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return typ->getAttribute(i)->isSubtypeOf(*TensorType::get()) &&
        typ->is_buffer(i);
  }
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = false;
};

struct TORCH_API AttributePolicy {
  using value_type = IValue;
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    return v;
  }
  static bool valid(const ClassTypePtr& typ, size_t i, const IValue& v) {
    return true;
  }
  static CONSTEXPR_EXCEPT_WIN_CUDA bool all_slots = true;
};

// take a Policy object, and make a version of it that returns the slot.
// along with the fully qualified name of that slot. This is used for the named_
// variants like named_parameters().
template <typename Policy>
struct NamedPolicy {
  using value_type = Named<typename Policy::value_type>;
  static value_type create(
      const std::vector<detail::SlotCursor>& cursors,
      IValue v) {
    std::string name;
    if (cursors.size() == 1) {
      name = (cursors.back().i_ == -1) ? "" : nameFragment(cursors.back());
    } else {
      std::ostringstream ss;
      for (const auto i : c10::irange(cursors.size())) {
        if (i > 0) {
          ss << ".";
        }
        ss << nameFragment(cursors[i]);
      }
      name = ss.str();
    }
    return value_type{std::move(name), Policy::create(cursors, std::move(v))};
  }
  static bool valid(const ClassTypePtr& t, size_t i, const IValue& v) {
    return Policy::valid(t, i, v);
  }
  static constexpr bool all_slots = Policy::all_slots;

 private:
  static std::string nameFragment(const detail::SlotCursor& f) {
    return f.module_.type()->getAttributeName(f.i_);
  }
};

} // namespace detail

TORCH_API bool& getInlineEverythingMode();

namespace script {
// We once had a `script::` namespace that was deleted. This is for backcompat
// of the public API; new code should not use this type alias.
using Module = ::torch::jit::Module;
using ExtraFilesMap = ::torch::jit::ExtraFilesMap;
} // namespace script

} // namespace jit
} // namespace torch

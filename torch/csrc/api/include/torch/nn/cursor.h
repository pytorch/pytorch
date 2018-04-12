#pragma once

#include <torch/tensor.h>

#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

// Forward declarations.
namespace torch { namespace nn {
class Module;
}} // namespace torch::nn

namespace torch {
namespace detail {
/// Provides hierarchical iteration support, with convenient iterator functions
/// like `map` or `find`.
template <typename T>
class CursorBase {
 public:
  // NOTE: This is a template class, but we explicitly instantiate it in the
  // .cpp file for every type necessary, so we can define it in the .cpp file.
  // Hooray!

  /// A `(key, value)` pair exposed by cursor iterators.
  struct Item {
    Item(const std::string& key_, T& module_);

    T& operator*();
    T* operator->();

    // These are references into a `Module`'s containers
    const std::string& key;
    T& value;
  };

  // Picks either `const_iterator` or `iterator` as the iterator type, depending
  // on whether `T` is const.
  using Iterator = typename std::conditional<
      std::is_const<T>::value,
      typename std::vector<Item>::const_iterator,
      typename std::vector<Item>::iterator>::type;

  // No need for a virtual destructor, as cursors are not intended to be used
  // polymorhpically (i.e. we are relying on non-virtual inheritance).

  // Note that these functions may only be called on lvalues (that's the
  // ampersand next to the function)! This prevents code like `auto iterator =
  // module.modules().begin()`, since `iterator` would be pointing to a `vector`
  // that gets destructed at the end of the expression. This is not a problem
  // for range loops, as they capture the range expression (the thing to the
  // right of the colon in `for (auto x : ...)`) before iteration.
  Iterator begin() & noexcept;
  Iterator end() & noexcept;

  /// Applies a function to every *value* available. The function should accept
  /// a single argument, that is a reference to the value type (e.g. `Module&`).
  template <typename Function>
  void apply(const Function& function) {
    for (auto module : *this) {
      function(*module);
    }
  }

  /// Applies a function to every *item* available. The function should accept
  /// two arguments, one taking a reference to the key type (always `const
  /// std::string&`) and the other taking a reference to the value type (e.g.
  /// `Module&`).
  template <typename Function>
  void apply_items(const Function& function) {
    for (auto module : *this) {
      function(module.key, *module);
    }
  }

  /// Applies a function to every *value* available, and stores the return value
  /// of the function into the iterator. The function should accept
  /// a single argument, that is a reference to the value type (e.g. `Module&`).
  template <typename Iterator, typename Function>
  void map(Iterator output_iterator, Function function) {
    for (auto module : *this) {
      *output_iterator = function(*module);
    }
  }

  /// Applies a function to every *value* available, and stores the return value
  /// of the function into the iterator. The function should accept
  /// two arguments, one taking a reference to the key type (always `const
  /// std::string&`) and the other taking a referen
  template <typename Iterator, typename Function>
  void map_items(Iterator output_iterator, Function function) {
    for (auto module : *this) {
      *output_iterator = function(module.key, *module);
    }
  }

  /// Attempts to find a value for the given `key`. If found, returns a pointer
  /// to the value. If not, returns a null pointer.
  T* find(const std::string& key) noexcept;

  /// Returns true if an item with the given `key` exists.
  bool contains(const std::string& key) noexcept;

  /// Counts the number of items available.
  size_t size() const noexcept;

 protected:
  std::vector<Item> items_;
};
} // namespace detail

namespace nn {

// Module cursors (`.modules()` and `.children()`)

class ModuleCursor : public detail::CursorBase<Module> {
 public:
  explicit ModuleCursor(
      Module& module,
      size_t maximum_depth = std::numeric_limits<size_t>::max());
};

class ConstModuleCursor : public detail::CursorBase<const Module> {
 public:
  explicit ConstModuleCursor(
      const Module& module,
      size_t maximum_depth = std::numeric_limits<size_t>::max());
};

// Parameter cursors (`.parameters()`)

class ParameterCursor : public detail::CursorBase<Tensor> {
 public:
  explicit ParameterCursor(Module& module);
};

class ConstParameterCursor : public detail::CursorBase<const Tensor> {
 public:
  explicit ConstParameterCursor(const Module& module);
};

// Buffer cursors (`.buffers()`)

class BufferCursor : public detail::CursorBase<Tensor> {
 public:
  explicit BufferCursor(Module& module);
};

class ConstBufferCursor : public detail::CursorBase<const Tensor> {
 public:
  explicit ConstBufferCursor(const Module& module);
};

} // namespace nn
} // namespace torch

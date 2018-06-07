#pragma once

#include <torch/csrc/autograd/variable.h>

#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

// Forward declarations.
namespace torch {
namespace detail {
template <typename T>
struct CursorCollector;
} // namespace detail
namespace nn {
class Module;
} // namespace nn
} // namespace torch

namespace torch {
namespace detail {
/// A cursor provides hierarchical iteration support, with convenient iterator
/// functions like `map` or `find`.
///
/// Fundamentally, cursors are similar to `std::map`, with `[]`, `.at()`,
/// `.size()` etc. When iterating over a cursor, use `.key()` to get the
/// associated string, and dereference the returned item or use `.value()` to
/// get the associated value (e.g. the variable, the module or the buffer).
///
/// Note that cursors *eagerly* collect items. So if you want to perform many
/// operations with a single cursor, it is better to store it in a local
/// variable. This also you means should not store iterators into a temporary
/// cursor, e.g. do not write `auto iterator = module.parameters().begin()`, as
/// the parameter cursor will die at the end of the expression.
///
/// A cursor's lifetime is bound to the lifetime of the module hierarchy into
/// which it points.
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

    const std::string key;
    T& value;
  };

  // Iterators are valid for the lifetime of the cursor.

  // Picks either `const_iterator` or `iterator` as the iterator type, depending
  // on whether `T` is const.
  using Iterator = typename std::conditional<
      std::is_const<T>::value,
      typename std::vector<Item>::const_iterator,
      typename std::vector<Item>::iterator>::type;

  CursorBase() = default;

  /// Constructs the `CursorBase` from a vector of items.
  explicit CursorBase(std::vector<Item>&& items);

  // No need for a virtual destructor, as cursors are not intended to be used
  // polymorhpically (i.e. we are relying on non-virtual inheritance).

  // Note that these functions may only be called on lvalues (that's the
  // ampersand next to the function)! This prevents code like `auto iterator =
  // module.modules().begin()`, since `iterator` would be pointing to a `vector`
  // that gets destructed at the end of the expression. This is not a problem
  // for range loops, as they capture the range expression (the thing to the
  // right of the colon in `for (auto x : ...)`) before iteration. This is
  // smart.
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

  /// Attempts to find a value for the given `key`. If found, returns a
  /// reference to the value. If not, throws an exception.
  T& at(const std::string& key);

  /// Equivalent to `at(key)`.
  T& operator[](const std::string& key);

  /// Returns true if an item with the given `key` exists.
  bool contains(const std::string& key) noexcept;

  /// Counts the number of items available.
  size_t size() const noexcept;

 protected:
  /// Helper struct to collect items.
  struct Collector;

  /// The (eagerly) collected vector of items.
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

class ParameterCursor : public detail::CursorBase<autograd::Variable> {
 public:
  explicit ParameterCursor(Module& module);
};

class ConstParameterCursor
    : public detail::CursorBase<const autograd::Variable> {
 public:
  explicit ConstParameterCursor(const Module& module);
};

// Buffer cursors (`.buffers()`)

class BufferCursor : public detail::CursorBase<autograd::Variable> {
 public:
  explicit BufferCursor(Module& module);
};

class ConstBufferCursor : public detail::CursorBase<const autograd::Variable> {
 public:
  explicit ConstBufferCursor(const Module& module);
};

} // namespace nn
} // namespace torch

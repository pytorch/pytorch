#pragma once

#include <torch/types.h>

#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

// Forward declarations confuse Doxygen
#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif // DOXYGEN_SHOULD_SKIP_THIS

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
  using ValueType = T;

  // NOTE: This is a template class, but we explicitly instantiate it in the
  // .cpp file for every type necessary, so we can define it in the .cpp file.
  // Hooray!

  /// A `(key, value)` pair exposed by cursor iterators.
  struct Item {
    Item(std::string key_, T& value_);

    T& operator*();
    const T& operator*() const;
    T* operator->();
    const T* operator->() const;

    const std::string key;
    T& value;
  };

  // Iterators are valid for the lifetime of the cursor.

  // Picks either `const_iterator` or `iterator` as the iterator type, depending
  // on whether `T` is const.
  using Iterator = typename std::vector<Item>::iterator;
  using ConstIterator = typename std::vector<Item>::const_iterator;

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
  ConstIterator begin() const& noexcept;

  Iterator end() & noexcept;
  ConstIterator end() const& noexcept;

  /// Applies a function to every *value* available. The function should accept
  /// a single argument, that is a reference to the value type (e.g. `Module&`).
  template <typename Function>
  void apply(const Function& function) {
    for (auto& item : items_) {
      function(*item);
    }
  }
  template <typename Function>
  void apply(const Function& function) const {
    for (auto& item : items_) {
      function(*item);
    }
  }

  /// Applies a function to every *item* available. The function should accept
  /// two arguments, one taking a reference to the key type (always `const
  /// std::string&`) and the other taking a reference to the value type (e.g.
  /// `Module&`).
  template <typename Function>
  void apply_items(const Function& function) {
    for (auto& item : items_) {
      function(item.key, item.value);
    }
  }
  template <typename Function>
  void apply_items(const Function& function) const {
    for (auto& item : items_) {
      function(item.key, item.value);
    }
  }

  /// Applies a function to every *value* available, and stores the return value
  /// of the function into the iterator. The function should accept
  /// a single argument, that is a reference to the value type (e.g. `Module&`).
  template <typename Iterator, typename Function>
  void map(Iterator output_iterator, Function function) {
    for (auto& item : items_) {
      *output_iterator++ = function(*item);
    }
  }
  template <typename Iterator, typename Function>
  void map(Iterator output_iterator, Function function) const {
    for (auto& item : items_) {
      *output_iterator++ = function(*item);
    }
  }

  /// Applies a function to every *value* available, and stores the return value
  /// of the function into the iterator. The function should accept
  /// two arguments, one taking a reference to the key type (always `const
  /// std::string&`) and the other taking a referen
  template <typename Iterator, typename Function>
  void map_items(Iterator output_iterator, Function function) {
    for (auto& item : items_) {
      *output_iterator++ = function(item.key, item.value);
    }
  }
  template <typename Iterator, typename Function>
  void map_items(Iterator output_iterator, Function function) const {
    for (auto& item : items_) {
      *output_iterator++ = function(item.key, item.value);
    }
  }

  /// Attempts to find a value for the given `key`. If found, returns a pointer
  /// to the value. If not, returns a null pointer.
  T* find(const std::string& key) noexcept;
  const T* find(const std::string& key) const noexcept;

  /// Attempts to find a value for the given `key`. If found, returns a
  /// reference to the value. If not, throws an exception.
  T& at(const std::string& key);
  const T& at(const std::string& key) const;

  /// Attempts to return the item at the given index. If the index is in range,
  /// returns a reference to the item. If not, throws an exception.
  Item& at(size_t index);

  /// Equivalent to `at(key)`.
  T& operator[](const std::string& key);
  const T& operator[](const std::string& key) const;

  /// Equivalent to `at(index)`.
  Item& operator[](size_t index);

  /// Returns true if an item with the given `key` exists.
  bool contains(const std::string& key) const noexcept;

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
  friend class ConstModuleCursor;
  explicit ModuleCursor(
      Module& module,
      size_t maximum_depth = std::numeric_limits<size_t>::max());
};

class ConstModuleCursor : public detail::CursorBase<const Module> {
 public:
  explicit ConstModuleCursor(
      const Module& module,
      size_t maximum_depth = std::numeric_limits<size_t>::max());

  /* implicit */ ConstModuleCursor(const ModuleCursor& cursor);
};

// Parameter cursors (`.parameters()`)

class ParameterCursor : public detail::CursorBase<Tensor> {
 public:
  friend class ConstParameterCursor;
  explicit ParameterCursor(Module& module);
};

class ConstParameterCursor : public detail::CursorBase<const Tensor> {
 public:
  explicit ConstParameterCursor(const Module& module);
  /* implicit */ ConstParameterCursor(const ParameterCursor& cursor);
};

// Buffer cursors (`.buffers()`)

class BufferCursor : public detail::CursorBase<Tensor> {
 public:
  friend class ConstBufferCursor;
  explicit BufferCursor(Module& module);
};

class ConstBufferCursor : public detail::CursorBase<const Tensor> {
 public:
  explicit ConstBufferCursor(const Module& module);
  /* implicit */ ConstBufferCursor(const BufferCursor& cursor);
};
} // namespace nn
} // namespace torch

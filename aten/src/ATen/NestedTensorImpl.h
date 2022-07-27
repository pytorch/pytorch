#pragma once
#include <ATen/MemoryOverlap.h>
#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/irange.h>

namespace at {
namespace native {

struct TORCH_API NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(at::Tensor buffer, at::Tensor nested_size_tensor);

  // Note: in current implementation,
  // empty nested tensor = no underlying tensor
  // nesting empty tensors is not considered empty nested tensor
  // both cases have `buffer_.numel() = 0`, though
  // TODO: for now, we use `nested_size_tensor_.dim()`
  // to determine whether a nested tensor is empty
  // when empty, `nested_size_tensor_.dim() = 0`
  // otherwise `nested_size_tensor_.dim() = 2`
  // maybe there is a better indicator for emptiness?

  // TODO: don't expose private implementation details like this; in
  // particular, resizing this tensor will mess up our dim() and
  // callers cannot fix it.
  const Tensor& get_nested_size_tensor() const {
    return nested_size_tensor_;
  }
  // TODO: don't expose private implementation details like this
  const Tensor& get_nested_stride_tensor() const {
    return nested_stride_tensor_;
  }
  // Returns nullopt if the ith dimension is irregular. The ith dimension
  // of a NestedTensor is regular if the unbound tensors match in
  // size at the (i-1)th dimension.
  c10::optional<int64_t> opt_size(int64_t d) const {
    d = at::maybe_wrap_dim(d, dim(), false);
    if (opt_sizes_[d] == -1) {
      return c10::nullopt;
    }
    return opt_sizes_[d];
  }

  int64_t size(int64_t d) const {
    c10::optional<int64_t> optional_size = this->opt_size(d);
    TORCH_CHECK(
        optional_size.has_value(),
        "Given dimension ",
        d,
        " is irregular and does not have a size.");
    return *optional_size;
  }

  const at::Tensor& get_buffer() const {
    return buffer_;
  }

 protected:
  const char* tensorimpl_type_name() const override;

  // TODO: numel_custom and is_contiguous_custom can be profitably overridden
  // with real implementations
  int64_t numel_custom() const override;
  bool is_contiguous_custom(MemoryFormat) const override;
  int64_t size_custom(int64_t d) const override {
    return this->size(d);
  }
  c10::SymInt sym_size_custom(int64_t d) const override {
    return c10::SymInt{this->size(d)};
  }
  IntArrayRef sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes() const override;
  IntArrayRef strides_custom() const override;

  // this one is real
  int64_t dim_custom() const override;

 private:
  // Must be called after any changes to our dim() to sync the state
  // to TensorImpl.
  void refresh_dim();

  at::Tensor buffer_;
  const at::Tensor nested_size_tensor_, nested_stride_tensor_;
  // NOTE: -1 here means the size is missing
  std::vector<int64_t> opt_sizes_;
};

inline NestedTensorImpl* get_nested_tensor_impl_or_null(
    const at::Tensor& tensor) {
  if (tensor.is_nested()) {
    return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
  }
  return nullptr;
}

inline NestedTensorImpl* get_nested_tensor_impl(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_nested(), "get_nested_tensor_impl requires a NestedTensor.");
  return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

// TODO: real implementation once we support strides.
inline bool nested_tensor_impl_is_contiguous(
    const NestedTensorImpl* nt,
    at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
  return memory_format == MemoryFormat::Contiguous;
}

inline const at::Tensor& get_nested_size_tensor(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_nested_size_tensor();
}

namespace impl {

// NOTE: For comparisons please use the map and reduce
// functions to define what you mean by equal, etc. on your own
// There can be ambiguity in the depth of comparison and
// even in the value (should it construct a new tree or
// return a single value).
template <typename T>
struct NestedNode {
  // NestedNode() : _is_leaf(false), _height(1) {}
  NestedNode() = delete;
  explicit NestedNode(std::vector<NestedNode<T>>&& children)
      : _is_leaf(false), _children(children), _height(1) {
    for (const auto& child : children) {
      if (child.height() + 1 > _height) {
        _height = child.height() + 1;
      }
    }
    // for (const auto& child : children) {
    //   TORCH_CHECK(
    //       child.height() == _height - 1,
    //       "internal error: expected a full tree.");
    // }
  }
  // NestedNode(NestedNode&) = delete;
  // NestedNode(const NestedNode&) = delete;
  // NestedNode& operator=(NestedNode) = delete;
  explicit NestedNode(T&& payload) : _is_leaf(true), _payload(payload), _height(0) {}
  inline bool is_leaf() const {
    return _is_leaf;
  }
  inline size_t degree() const {
    return _children.size();
  }
  inline int64_t height() const {
    return _height;
  }
  inline const std::vector<NestedNode<T>> unbind() const {
    return _children;
  }
  inline NestedNode<T> children(size_t i) const {
    return _children[i];
  }
  inline const T& payload() const {
    return _payload;
  }
  inline T& payload() {
    return _payload;
  }

 private:
  bool _is_leaf;
  std::vector<NestedNode<T>> _children;
  // TODO: Make this const?
  // _VariableNode _variable_node;
  T _payload;
  int64_t _height;
};

using TensorNode = NestedNode<at::Tensor>;

template <class F, class A, class TypeList>
class _map;

template <class F, class A, class... Args>
class _map<F, A, c10::guts::typelist::typelist<Args...>> {
 public:
  // NOTE: We must move F to avoid copying objects if it is a lambda with
  // captures.
  static NestedNode<A> function(
      F&& fn,
      const NestedNode<Args>&... nested_node) {
    size_t degree = 0;
    bool all_leaf = true;
    c10::guts::tuple_map(
        std::forward_as_tuple(nested_node...), [&all_leaf, &degree](auto n) {
          all_leaf = all_leaf && (n.is_leaf());
          if (degree > 1 && n.degree() > 1) {
            TORCH_CHECK(degree == n.degree(), "NestedNodes don't broadcast.");
          }
          if (n.degree() > degree) {
            degree = n.degree();
          }
          return nullptr;
        });
    if (all_leaf) {
      return NestedNode<A>(std::forward<F>(fn)(nested_node.payload()...));
    }
    std::vector<NestedNode<A>> result;
    for (size_t i = 0; i < degree; i++) {
      std::tuple<NestedNode<Args>...> children = c10::guts::tuple_map(
          std::forward_as_tuple(nested_node...), [&i](auto a) {
            static_assert(
                c10::guts::is_instantiation_of<NestedNode, decltype(a)>::value,
                "Internal error.");
            if (a.is_leaf()) {
              return a;
            }
            if (a.degree() == 1 && a.height() > 0) {
              return a.children(0);
            }
            TORCH_CHECK(a.degree() > 0, "Internal assert.");
            return a.children(i);
          });
      // TODO: Due to the experiences with to_vector and the inversion I'm a bit
      // wary of apply but I haven't been able to reproduce the  argument
      // inversion behavior in other contexts.
      c10::guts::apply(
          [&result, &fn](NestedNode<Args>... filtered) {
            result.emplace_back(function(std::forward<F>(fn), filtered...));
          },
          std::move(children));
    }
    return NestedNode<A>(std::move(result));
  }
};

// NOTE: Assuming all NestedNodes have same shape.
// TODO: Add check
// TODO: Add static assert to verify lambda arguments match nested_node types
// TODO: Do we want broadcasting?
template <class F, class... B>
static inline NestedNode<
    typename c10::guts::infer_function_traits<F>::type::return_type>
map(F&& fn, const NestedNode<B>&... nested_node) {
  return _map<
      F,
      typename c10::guts::infer_function_traits<F>::type::return_type,
      typename c10::guts::infer_function_traits<F>::type::parameter_types>::
      function(std::forward<F>(fn), nested_node...);
}

template <>
inline TensorNode get_nested_tensor_structure(at::Tensor tensor) {
  auto nt_impl = get_nested_tensor_impl_or_null(tensor);
  if (nt_impl == nullptr) {
    return TensorNode(std::move(tensor));
  }
  return get_nested_tensor_impl(tensor)->get_structure();
    return std::get<0>(torch::nested_tensor::impl::build_structure(
        impl->get_buffer().reshape({-1}),
        impl->get_nested_size_tensor(),
        impl->get_nested_stride_tensor()));
}


}

template <class F, class... A>
inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  return wrap_tensor_node(
      map(std::forward<F>(fn), get_nested_tensor_structure(a)...));
}

} // namespace native
} // namespace at

#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <ATen/NestedTensorImpl.h>

#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// TODO: cache this and only do it once per NestedTensor
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_size_tensor) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_size_tensor));
}


TORCH_API std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt);

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(const Tensor& t, double padding, OptionalIntArrayRef output_size);

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
      : _is_leaf(false), _children(children) {
    for (const auto& child : children) {
      TORCH_CHECK(child.is_leaf(), "Expected children to be leafs. No support for multiple nested dimensions yet.");
    }
  }
  // NestedNode(NestedNode&) = delete;
  // NestedNode(const NestedNode&) = delete;
  // NestedNode& operator=(NestedNode) = delete;
  explicit NestedNode(T&& payload) : _is_leaf(true), _payload(payload) {}
  inline bool is_leaf() const {
    return _is_leaf;
  }
  inline size_t degree() const {
    return _children.size();
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
            if (a.degree() == 1 && !a.is_leaf()) {
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

inline TensorNode get_nested_tensor_structure(at::Tensor tensor) {
  if (get_nested_tensor_impl_or_null(tensor) == nullptr) {
    return TensorNode(std::move(tensor));
  }
  std::vector<TensorNode> structure;
  for (auto tensor_i : tensor.unbind()) {
    structure.push_back(TensorNode(std::move(tensor_i)));
  }
  return TensorNode(std::move(structure));
}

inline Tensor wrap_tensor_node(TensorNode tensor_node,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  TORCH_CHECK(!tensor_node.is_leaf(), "Expected TensorNode to wrap a list of Tensors.");
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  if (tensor_node.degree() == 0) {
    return wrap_buffer(ones({0}, dtype, layout, device), ones({}));
  }
  std::vector<Tensor> sizes;
  std::vector<Tensor> flat_tensors;
  for (const auto i : c10::irange(tensor_node.degree())) {
    // TODO: Remove call to contiguous once we support strides.
    flat_tensors.push_back(tensor_node.children(i).payload().reshape(-1).contiguous());
    sizes.push_back(tensor(c10::IntArrayRef(tensor_node.children(i).payload().sizes())));
  }

  TensorOptions options = flat_tensors[0].options().merge_in(options_);

  return wrap_buffer(
      at::cat(flat_tensors).to(options), at::native::stack(sizes));
}


}

template <class F, class... A>
inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  return wrap_tensor_node(impl::map(std::forward<F>(fn), impl::get_nested_tensor_structure(a)...),
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
}

} // namespace native
} // namespace at

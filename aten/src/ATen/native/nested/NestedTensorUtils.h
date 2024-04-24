#pragma once

#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ones_native.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/stack_native.h>
#include <ATen/ops/tensor.h>
#endif

#include <utility>
#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// The following functions are used to construct nested tensors from buffers and
// metadata.

inline at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_sizes) {
  TORCH_CHECK(
      buffer.dim() == 1,
      "Expected given buffer to be 1dim, but got ",
      buffer.dim(),
      " instead.");
  TORCH_CHECK(
      buffer.is_contiguous(), "Expected given buffer to be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_sizes));
}

// TODO: Figure out if we need a non-moving wrap_buffer()
inline at::Tensor wrap_buffer(
    at::Tensor buffer,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      std::move(nested_sizes),
      std::move(nested_strides),
      std::move(storage_offsets));
}

inline at::Tensor get_buffer(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_buffer();
}

/**
 * Create a new nested tensor that is a view of a base nested tensor
 *
 * create_view_tensor calls a specialized constructor that copys the
 * the keys from base onto the new view tensor being created.
 * The storage is shared between the base and the returned view tensor
 *
 * All callers of this helper must:
 * - Only return a view of the input
 * - Must be explicit and define a derivative
 *
 * @param base Base tensor to construct view from.
 * @param nested_sizes View tensors' sizes.
 * @param nested_strides View tensors' strides.
 * @param storage_offsets View tensors' offsets.
 * @return A newly constructed view tensor
 */
inline at::Tensor create_nested_view_tensor(
    const at::Tensor& base,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets) {
  TORCH_INTERNAL_ASSERT(
      base.is_nested(),
      "This function can only be used to create nested tensor views");
  TORCH_INTERNAL_ASSERT(
      c10::impl::tls_local_dispatch_key_set().excluded_.has(
          c10::DispatchKey::AutogradFunctionality),
      "Creating a non differentiable nested tensor view in a CompositeImplicit function is not allowed.");
  return at::detail::make_tensor<NestedTensorImpl>(
      c10::TensorImpl::VIEW,
      base,
      nested_sizes,
      nested_strides,
      storage_offsets);
}
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Helper functions for getting information about a nested tensor's shape.

int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

// The sizes of the underlying tensors
inline std::vector<IntArrayRef> NestedTensor_get_sizes(
    const NestedTensorImpl* self_ptr) {
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> sizes(ntensors);
  if (ntensors == 0) {
    return sizes;
  }
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  int64_t orig_dim = sizemat.size(1);
  // nesting scalars has empty sizes
  if (orig_dim == 0) {
    return sizes;
  }
  const int64_t* sizemat_ptr = sizemat.data_ptr<int64_t>();

  for (const auto i : c10::irange(ntensors)) {
    sizes[i] = IntArrayRef(sizemat_ptr, sizemat_ptr + orig_dim);
    sizemat_ptr += orig_dim;
  }
  return sizes;
}

TORCH_API std::vector<int64_t> NestedTensor_get_max_size(
    const NestedTensorImpl& nt);

std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(
    const Tensor& sizes);

inline std::vector<IntArrayRef> NestedTensor_get_sizes(const at::Tensor& self) {
  const NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  return NestedTensor_get_sizes(self_ptr);
}
// The strides of the underlying tensors
inline std::vector<IntArrayRef> NestedTensor_get_strides(
    const NestedTensorImpl* self_ptr) {
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> strides(ntensors);
  if (ntensors == 0) {
    return strides;
  }
  const Tensor& stridemat = self_ptr->get_nested_strides();
  int64_t orig_dim = stridemat.size(1);
  // nesting scalars has empty strides
  if (orig_dim == 0) {
    return strides;
  }
  const int64_t* stridemat_ptr = stridemat.data_ptr<int64_t>();
  for (const auto i : c10::irange(ntensors)) {
    strides[i] = IntArrayRef(stridemat_ptr, stridemat_ptr + orig_dim);
    stridemat_ptr += orig_dim;
  }
  return strides;
}

inline std::vector<IntArrayRef> NestedTensor_get_strides(
    const at::Tensor& self) {
  const NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  return NestedTensor_get_strides(self_ptr);
}

inline void check_numel_equals_buffer_size(const at::Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  TORCH_CHECK(
      self.numel() == static_cast<int64_t>(self_impl->get_buffer_size()),
      "Number of elements in nested tensor must match number of elements in buffer.");
}

inline void check_numel_equals_buffer_size(const NestedTensorImpl* self_ptr) {
  TORCH_CHECK(
      self_ptr->numel() == static_cast<int64_t>(self_ptr->get_buffer_size()),
      "Number of elements in nested tensor must match number of elements in buffer.");
}

// Helper function to get size / stride / offset for a nested/normal tensor.
inline IntArrayRef get_size_for_index(const Tensor& tensor, int i) {
  if (tensor.is_nested()) {
    std::vector<IntArrayRef> tensor_sizes =
        NestedTensor_get_sizes(get_nested_tensor_impl(tensor));
    return tensor_sizes[i];
  } else {
    return tensor.sizes().slice(1);
  }
}

inline IntArrayRef get_stride_for_index(const Tensor& tensor, int i) {
  if (tensor.is_nested()) {
    std::vector<IntArrayRef> tensor_strides =
        NestedTensor_get_strides(get_nested_tensor_impl(tensor));
    return tensor_strides[i];
  } else {
    return tensor.strides().slice(1);
  }
}

inline int64_t get_offset_for_index(const Tensor& tensor, int i) {
  if (tensor.is_nested()) {
    int64_t* offsets_ptr = get_nested_tensor_impl(tensor)
                               ->get_storage_offsets()
                               .data_ptr<int64_t>();
    return offsets_ptr[i];

  } else {
    int64_t offset = tensor.storage_offset();
    return offset + tensor.strides()[0] * i;
  }
}
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Data structures and functions for generically applying a function on a nested
// tensor.
namespace impl {

template <typename T>
struct NestedNode {
  NestedNode() = delete;
  explicit NestedNode(std::vector<T>&& children)
      : _is_leaf(false), _children(children) {}
  explicit NestedNode(TensorList children)
      : _is_leaf(false), _children(children.vec()) {}
  // NestedNode(NestedNode&) = delete;
  // NestedNode(const NestedNode&) = delete;
  // NestedNode& operator=(NestedNode) = delete;
  explicit NestedNode(T payload)
      : _is_leaf(true), _payload(std::move(payload)) {}
  inline bool is_leaf() const {
    return _is_leaf;
  }
  inline size_t degree() const {
    return _children.size();
  }
  inline const std::vector<T> unbind() const {
    return _children;
  }
  inline T children(size_t i) const {
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
  std::vector<T> _children;
  T _payload;
};

using TensorNode = NestedNode<at::Tensor>;

template <class F, class A, class TypeList>
class _map;

template <class F, class A, class... Args>
class _map<F, A, c10::guts::typelist::typelist<Args...>> {
 public:
  static A function_one(F&& fn, const Args&... nested_node) {
    return std::forward<F>(fn)(nested_node...);
  }
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
            TORCH_CHECK(
                degree == n.degree(), "NestedNodes must match in degree.");
          }
          if (n.degree() > degree) {
            degree = n.degree();
          }
          return nullptr;
        });
    // All NestedNodes just wrap regular objects.
    if (all_leaf) {
      return NestedNode<A>(std::forward<F>(fn)(nested_node.payload()...));
    }
    // Some NestedNodes wrap regular Tensors, some NestedTensors and some other
    // types.
    std::vector<A> result;
    for (size_t i = 0; i < degree; i++) {
      std::tuple<Args...> children = c10::guts::tuple_map(
          std::forward_as_tuple(nested_node...), [&i](auto a) {
            static_assert(
                c10::guts::is_instantiation_of<NestedNode, decltype(a)>::value,
                "Internal error.");
            // Broadcast regular arguments across NestedTensor constituents.
            // This could be a Tensor, integer or anything else really.
            if (a.is_leaf()) {
              return a.payload();
            }
            // Broadcast NestedTensors with one constituent.
            if (a.degree() == 1 && !a.is_leaf()) {
              return a.children(0);
            }
            TORCH_CHECK(a.degree() > 0, "Internal assert.");
            return a.children(i);
          });
      c10::guts::apply(
          [&result, &fn](Args... filtered) {
            result.emplace_back(function_one(std::forward<F>(fn), filtered...));
          },
          std::move(children));
    }
    return NestedNode<A>(std::move(result));
  }
};

// TODO: Add static assert to verify lambda arguments match nested_node types
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
  return TensorNode(tensor.unbind());
}

inline Tensor wrap_tensor_node(
    TensorNode tensor_node,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  TORCH_CHECK(
      !tensor_node.is_leaf(), "Expected TensorNode to wrap a list of Tensors.");
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  if (tensor_node.degree() == 0) {
    return wrap_buffer(ones({0}, dtype, layout, device), ones({}));
  }

  // Fast path: if all tensors are on CPU, have contiguous memory, and the same
  // dtype, copying can be done much faster.
  bool all_tensors_cpu = true;
  bool all_tensors_contiguous = true;
  bool all_tensors_same_dtype = true;
  auto first_dtype = tensor_node.children(0).dtype();
  std::vector<long> start_offsets(tensor_node.degree());
  start_offsets[0] = 0;
  long total_size = 0;
  for (const auto i : c10::irange(tensor_node.degree())) {
    all_tensors_cpu = all_tensors_cpu && tensor_node.children(i).is_cpu();
    all_tensors_contiguous =
        all_tensors_contiguous && tensor_node.children(i).is_contiguous();
    all_tensors_same_dtype = all_tensors_same_dtype &&
        (first_dtype == tensor_node.children(i).dtype());
    if (!(all_tensors_cpu && all_tensors_contiguous &&
          all_tensors_same_dtype)) {
      break;
    }
    if (i > 0) {
      start_offsets[i] =
          start_offsets[i - 1] + tensor_node.children(i - 1).numel();
    }
    total_size += tensor_node.children(i).numel();
  }

  TensorOptions options;
  Tensor nt_buffer, nt_sizes;
  if (all_tensors_cpu && all_tensors_contiguous && all_tensors_same_dtype) {
    nt_buffer = at::empty({total_size}, tensor_node.children(0).options());
    nt_sizes = at::empty(
        {static_cast<long>(tensor_node.degree()),
         static_cast<long>(tensor_node.children(0).sizes().size())},
        TensorOptions().dtype(kLong));
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        c10::typeMetaToScalarType(first_dtype),
        "create_nt_buffer",
        [&]() {
          at::parallel_for(
              0, tensor_node.degree(), 1, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                  // Only try copying memory if there is more than 0 elements
                  // for a certain tensor
                  if (tensor_node.children(i).numel() > 0) {
                    memcpy(
                        nt_buffer.mutable_data_ptr<scalar_t>() + start_offsets[i],
                        tensor_node.children(i).const_data_ptr<scalar_t>(),
                        tensor_node.children(i).numel() * sizeof(scalar_t));
                  }
                }
              });
        });
    long sizes_offset = 0;
    for (size_t i = 0; i < tensor_node.degree(); ++i) {
      auto tensor_sizes = tensor_node.children(i).sizes();
      for (int64_t tensor_size : tensor_sizes) {
        nt_sizes.mutable_data_ptr<int64_t>()[sizes_offset++] = tensor_size;
      }
    }
    options = nt_buffer.options().merge_in(options_);
  } else { // Slow path
    std::vector<Tensor> flat_tensors;
    std::vector<Tensor> sizes;
    for (const auto i : c10::irange(tensor_node.degree())) {
      flat_tensors.push_back(tensor_node.children(i).reshape(-1).contiguous());
      sizes.push_back(
          tensor(c10::IntArrayRef(tensor_node.children(i).sizes())));
    }
    options = flat_tensors[0].options().merge_in(options_);
    nt_buffer = at::cat(flat_tensors);
    nt_sizes = at::native::stack(sizes);
  }

  return wrap_buffer(nt_buffer.to(options), nt_sizes);
}

} // namespace impl

// This function is meant to ease rapid operator coverage for
// NestedTensor kernels. It is not meant to be efficient. Use it judiciously.
template <class F, class... A>
inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  return wrap_tensor_node(
      impl::map(std::forward<F>(fn), impl::get_nested_tensor_structure(a)...),
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
}

} // namespace native
} // namespace at

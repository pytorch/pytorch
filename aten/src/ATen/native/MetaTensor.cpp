#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

// Will be promoted to a public API later, but not now
Tensor empty_meta(
  IntArrayRef size,
  const TensorOptions& options_,
  c10::optional<c10::MemoryFormat> optional_memory_format
) {
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  TensorOptions options = options_.merge_in(TensorOptions().memory_format(optional_memory_format));

  // TODO: deduplicate this logic with empty_cpu

  auto dtype = options.dtype();
  auto device = options.device();
  auto tensor = detail::make_tensor<TensorImpl>(
    // NB: We include the computed dispatch key, not because it will actually
    // participate in dispatch, but so that tests like is_sparse/is_cuda
    // give the correct result (a CUDA meta tensor "is cuda").  If we don't
    // like this, remove the computeDispatchKey line
    DispatchKeySet{DispatchKey::Meta, computeDispatchKey(options)},
    dtype,
    device
  );
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  return tensor;
}

} // namespace native
} // namespace at

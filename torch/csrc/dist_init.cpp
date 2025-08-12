#include <ATen/ATen.h>
#include <torch/library.h>

// Unlike files in torch/csrc/distributed, this is UNCONDITIONALLY compiled
// even when we don't have distributed, so that these definitions always exist
// even if they don't have implementations

namespace {

at::Tensor one_shot_all_reduce_meta(
    const at::Tensor& input,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  return at::empty_like(input);
}

at::Tensor one_shot_all_reduce_copy_meta(
    const at::Tensor& symm_buffer,
    const at::Tensor& local_input,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  return at::empty_like(local_input);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "multimem_all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)");
  m.def(
      "multimem_one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor");
  m.def(
      "multimem_one_shot_all_reduce_out(Tensor input, str reduce_op, str group_name, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "multimem_all_gather_out(Tensor input, str group_name, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor");
  m.def(
      "one_shot_all_reduce_out(Tensor input, str reduce_op, str group_name, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "one_shot_all_reduce_copy(Tensor symm_buffer, Tensor local_input, str reduce_op, str group_name) -> Tensor");
  m.def(
      "one_shot_all_reduce_copy_out(Tensor symm_buffer, Tensor local_input, str reduce_op, str group_name, Tensor(a!) out) -> Tensor(a!)");

  m.def(
      "two_shot_all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)");

  // note this implementation also modified the input tensor
  m.def(
      "two_shot_all_reduce_out(Tensor(a!) input, str reduce_op, str group_name, Tensor(b!) output) -> Tensor(b!)");

  // note this implementation also modified the input tensor
  m.def(
      "reduce_scatter_out(Tensor(a!) input, str group_name, bool split_last_dim, Tensor(b!) output) -> Tensor(b!)");

  // An mm that supports consuming asynchronous input. It guarantees the
  // following rasterization order, and that the corresponding signal arrives
  // before an input chunk is consumed.
  //
  // num_chunks = a_chunks_signals.numel()
  // for chunk_idx in range(a_chunk_pivot, num_chunks + a_chunk_pivot):
  //     chunk_idx = chunk_idx % num_chunks
  //     wait_signal(a_chunk_signals, chunk_idx)
  //     # Compute output tiles that consumes the input chunk
  m.def(
      "_async_input_mm(Tensor a, Tensor b, Tensor a_chunk_signals, int a_chunk_pivot) -> Tensor");
  m.def(
      "stream_write_value32_(Tensor(a!) input, int offset, int val) -> Tensor(a!)");
  m.def(
      "memset32_(Tensor(a!) input, int offset, int val, int count) -> Tensor(a!)");

  m.def("nvshmem_put(Tensor(a!) tensor, int peer) -> ()");
  m.def("nvshmem_get(Tensor(a!) tensor, int peer) -> ()");
  m.def("nvshmem_broadcast(Tensor(a!) input, str group_name) -> Tensor(a!)");
  m.def(
      "nvshmem_all_to_all(Tensor input, Tensor(a!) out, str group_name) -> Tensor(a!)");
  m.def(
      "all_to_all_vdev(Tensor input, Tensor(a!) out, Tensor(a!) in_out_splits, str group_name) -> Tensor(a!)");
  m.def(
      "all_to_all_vdev_2d(Tensor input, Tensor(a!) out, Tensor in_splits, Tensor(a!) out_splits_offsets, str group_name, int? major_align=None) -> ()");
  m.def(
      "all_to_all_vdev_2d_offset(Tensor input, Tensor(a!) out, Tensor in_splits_offsets, Tensor(a!) out_splits_offsets, str group_name) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, Meta, m) {
  m.impl("one_shot_all_reduce", one_shot_all_reduce_meta);
  m.impl("one_shot_all_reduce_copy", one_shot_all_reduce_copy_meta);
}

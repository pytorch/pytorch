#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace {

using namespace c10d::symmetric_memory;

static bool is_finalizing_ = false;

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class AllocatorMap {
 public:
  AllocatorMap(const AllocatorMap&) = delete;
  AllocatorMap& operator=(const AllocatorMap&) = delete;
  static AllocatorMap& get() {
    static AllocatorMap instance;
    return instance;
  }

  void register_allocator(
      c10::DeviceType device_type,
      c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
    map_[device_type] = std::move(allocator);
  }

  void register_availability(
      const std::string& name,
      c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
    avail_map_[name] = std::move(allocator);
  }

  void set_backend(const std::string& name) {
    auto it = avail_map_.find(name);
    TORCH_CHECK(
        it != avail_map_.end(),
        "SymmetricMemory does not find allocation backend ",
        name);
    auto device_type = it->second->supported_device_type();
    // Check if the existing one is already the one desired.
    auto existing = map_.find(device_type);
    if (existing != map_.end()) {
      if (existing->second->name() == name) {
        // The existing one is the same as the desired one. No need to change.
        return;
      }
      TORCH_CHECK(!in_use_, "Backend can not be changed after use.");
    }
    register_allocator(device_type, it->second);
  }

  std::optional<std::string> get_backend(c10::DeviceType device_type) {
    auto it = map_.find(device_type);
    if (it == map_.end()) {
      return std::nullopt;
    }
    return it->second->name();
  }

  c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
      c10::DeviceType device_type) {
    auto it = map_.find(device_type);
    TORCH_CHECK(
        it != map_.end(),
        "SymmetricMemory does not support device type ",
        device_type);
    in_use_ = true;
    return it->second;
  }

  bool has_allocator(c10::DeviceType device_type) {
    auto it = map_.find(device_type);
    return it != map_.end();
  }

  ~AllocatorMap() {
    is_finalizing_ = true;
  }

 private:
  AllocatorMap() = default;

  std::unordered_map<
      c10::DeviceType,
      c10::intrusive_ptr<SymmetricMemoryAllocator>>
      map_;

  // For backends to register availability.
  // This registration is at static time. Therefore, it is expected that the
  // derived `SymmetricMemoryAllocator` classes do not have backend-specific
  // initialization in constructor (in case it is not selected).
  std::unordered_map<
      std::string, // backend name "NVSHMEM", "CUDA", "NCCL", etc.
      c10::intrusive_ptr<SymmetricMemoryAllocator>>
      avail_map_;

  bool in_use_ = false;
};

static std::unordered_map<std::string, GroupInfo> group_info_map{};

// Data structures for tracking persistent allocations
static std::unordered_map<uint64_t, void*> alloc_id_to_dev_ptr{};
static std::unordered_map<uint64_t, c10::weak_intrusive_ptr<c10::StorageImpl>>
    alloc_id_to_storage{};

static at::Tensor empty_strided_p2p_persistent(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::optional<std::string>& group_name,
    uint64_t alloc_id) {
  // Make the allocation fails if a previous allocation with the same alloc_id
  // is still active.
  auto storage = alloc_id_to_storage.find(alloc_id);
  if (storage != alloc_id_to_storage.end() && storage->second.use_count() > 0) {
    TORCH_CHECK(
        false,
        "SymmetricMemory::empty_strided_p2p_persistent: ",
        "can not allocate with alloc_id == ",
        alloc_id,
        " because a previous allocation with the same alloc_id "
        "is still active.");
  }

  const size_t numel = std::accumulate(
      size.begin(),
      size.end(),
      size_t(1),
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<size_t>());
  const size_t element_size = c10::elementSize(dtype);
  const size_t alloc_size = numel * element_size;

  auto allocator = get_allocator(device.type());
  void* dev_ptr = nullptr;
  if (alloc_id_to_dev_ptr.find(alloc_id) != alloc_id_to_dev_ptr.end()) {
    dev_ptr = alloc_id_to_dev_ptr[alloc_id];
    TORCH_CHECK(
        alloc_size == allocator->get_alloc_size(dev_ptr),
        "SymmetricMemory::empty_strided_p2p_persistent: ",
        "requested allocation size (",
        alloc_size,
        ") is different from the size of a previous allocation ",
        "with the same alloc_id ",
        allocator->get_alloc_size(dev_ptr));
  } else {
    dev_ptr = allocator->alloc(alloc_size, device.index(), group_name);
    alloc_id_to_dev_ptr[alloc_id] = dev_ptr;
  }

  auto options = at::TensorOptions().dtype(dtype).device(device);
  auto allocated = at::from_blob(dev_ptr, size, stride, options);

  // Track the allocation's activeness
  alloc_id_to_storage.erase(alloc_id);
  alloc_id_to_storage.emplace(
      alloc_id, allocated.storage().getWeakStorageImpl());
  return allocated;
}

} // namespace

namespace c10d::symmetric_memory {

bool is_finalizing() {
  return is_finalizing_;
}

void register_allocator(
    c10::DeviceType device_type,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
  return AllocatorMap::get().register_allocator(
      device_type, std::move(allocator));
}

void register_availability(
    const std::string& name,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
  return AllocatorMap::get().register_availability(name, std::move(allocator));
}

void set_backend(const std::string& name) {
  return AllocatorMap::get().set_backend(name);
}

std::optional<std::string> get_backend(c10::Device device) {
  return AllocatorMap::get().get_backend(device.type());
}

bool has_allocator(c10::DeviceType device_type) {
  return AllocatorMap::get().has_allocator(device_type);
}

c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
    c10::DeviceType device_type) {
  return AllocatorMap::get().get_allocator(device_type);
}

void set_group_info(
    const std::string& group_name,
    int rank,
    int world_size,
    c10::intrusive_ptr<Store> store) {
  TORCH_CHECK(group_info_map.find(group_name) == group_info_map.end());
  GroupInfo group_info;
  group_info.rank = rank;
  group_info.world_size = world_size;
  group_info.store = std::move(store);
  group_info_map.emplace(group_name, std::move(group_info));
}

GroupInfo& get_group_info(const std::string& group_name) {
  TORCH_CHECK(
      group_info_map.find(group_name) != group_info_map.end(),
      "get_group_info: no group info associated with the group name ",
      group_name);
  return group_info_map[group_name];
}

at::Tensor empty_strided_p2p(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::optional<std::string>& group_name,
    std::optional<uint64_t> alloc_id) {
  if (alloc_id.has_value()) {
    return empty_strided_p2p_persistent(
        size, stride, dtype, device, group_name, *alloc_id);
  }
  const size_t numel = std::accumulate(
      size.begin(),
      size.end(),
      size_t(1),
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<size_t>());
  const size_t element_size = c10::elementSize(dtype);
  const size_t alloc_size = numel * element_size;

  auto allocator = get_allocator(device.type());
  void* dev_ptr = allocator->alloc(alloc_size, device.index(), group_name);

  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::from_blob(
      dev_ptr,
      size,
      stride,
      [allocator = std::move(allocator)](void* ptr) { allocator->free(ptr); },
      options);
}

TORCH_API c10::intrusive_ptr<SymmetricMemory> rendezvous(
    const at::Tensor& tensor,
    const std::optional<std::string>& group_name) {
  auto allocator = get_allocator(tensor.device().type());
  return allocator->rendezvous(tensor.storage().data_ptr().get(), group_name);
}

TORCH_API bool has_multicast_support(
    c10::DeviceType device_type,
    int device_idx) {
  if (!has_allocator(device_type)) {
    return false;
  } else {
    auto allocator = get_allocator(device_type);
    return allocator->has_multicast_support(device_idx);
  }
}

// MemPool Support

// A map from device type to allocator for MemPool.
// TODO: Consolidate with `AllocatorMap` above.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class MemPoolAllocatorMap {
 public:
  MemPoolAllocatorMap(const MemPoolAllocatorMap&) = delete;
  MemPoolAllocatorMap& operator=(const MemPoolAllocatorMap&) = delete;
  static MemPoolAllocatorMap& get() {
    static MemPoolAllocatorMap instance;
    return instance;
  }

  // Register allocator for MemPool given device type
  void register_mempool_allocator(
      c10::DeviceType device_type,
      std::shared_ptr<c10::Allocator> allocator) {
    mempool_allocators_[device_type] = std::move(allocator);
  }

  // Get allocator for MemPool given device
  std::shared_ptr<c10::Allocator> get_mempool_allocator(c10::Device device) {
    auto it = mempool_allocators_.find(device.type());
    if (it == mempool_allocators_.end()) {
      TORCH_CHECK(
          false,
          "SymmetricMemory MemPool did not find backend for device type ",
          device.type());
    }
    return it->second;
  }

 private:
  MemPoolAllocatorMap() = default;

  std::unordered_map<c10::DeviceType, std::shared_ptr<c10::Allocator>>
      mempool_allocators_;
};

// Register allocator for MemPool given device type
C10_EXPORT void register_mempool_allocator(
    c10::DeviceType device_type,
    std::shared_ptr<c10::Allocator> allocator) {
  return MemPoolAllocatorMap::get().register_mempool_allocator(
      device_type, std::move(allocator));
}

// Get allocator for MemPool given device
TORCH_API std::shared_ptr<c10::Allocator> get_mempool_allocator(
    c10::Device device) {
  return MemPoolAllocatorMap::get().get_mempool_allocator(device);
}

// Helper function:
// Calculate the number of bytes of a tensor given its shape and dtype
static inline size_t nbytes_of(c10::IntArrayRef sizes, c10::ScalarType dtype) {
  const auto numel = std::accumulate(
      sizes.begin(), sizes.end(), static_cast<size_t>(1), std::multiplies<>());
  return numel * c10::elementSize(dtype);
}

// Helper function:
// Get the buffer pointer for a peer at a given offset
static at::Tensor get_buffer_at_byte_offset(
    SymmetricMemory* handle,
    int peer,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    size_t offset_bytes) {
  TORCH_CHECK(
      peer >= 0 && peer < handle->get_world_size(),
      "Invalid peer rank: ",
      peer);
  auto peer_ptr = handle->get_buffer_ptrs()[peer];
  TORCH_CHECK(
      peer_ptr != nullptr,
      "Cannot get buffer across nodes, my rank: ",
      handle->get_rank(),
      ", peer: ",
      peer);
  const size_t tensor_bytes = nbytes_of(sizes, dtype);
  const auto req_size = offset_bytes + tensor_bytes;
  const auto buffer_size = handle->get_buffer_size();
  TORCH_CHECK(
      req_size <= buffer_size,
      "SymmetricMemory::get_buffer: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      buffer_size,
      " bytes)");
  auto data_ptr = reinterpret_cast<uint8_t*>(peer_ptr) + offset_bytes;
  auto device = handle->get_device();
  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::for_blob(data_ptr, sizes)
      .options(options)
      .target_device(device)
      .make_tensor();
}

// Implementation of SymmetricMemory APIs common to all backends

at::Tensor SymmetricMemory::get_buffer(
    int rank,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    int64_t storage_offset) {
  // storage_offset is in element, convert to byte
  const auto offset_bytes = storage_offset * c10::elementSize(dtype);
  return get_buffer_at_byte_offset(this, rank, sizes, dtype, offset_bytes);
}

at::Tensor SymmetricMemory::get_remote_tensor(
    int peer,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype) {
  return get_buffer_at_byte_offset(this, peer, sizes, dtype, get_offset());
}

at::Tensor SymmetricMemory::get_signal_pad(
    int rank,
    c10::IntArrayRef sizes,
    std::optional<c10::ScalarType> dtype,
    int64_t storage_offset) {
  // If the dtype is unspecified, default it to UInt32, as it
  // is the most common type for signaling purposes.
  if (!dtype.has_value()) {
    dtype = c10::ScalarType::UInt32;
  }

  // If the shape is unspecified, treat the signal pad as a 1d tensor.
  const auto element_size = c10::elementSize(*dtype);
  const auto signal_pad_size = get_signal_pad_size();
  std::vector<int64_t> shape;
  if (!sizes.empty()) {
    shape = sizes.vec();
  } else {
    shape.push_back(static_cast<int64_t>(signal_pad_size / element_size));
  }

  const auto req_pad_bytes = nbytes_of(shape, *dtype);
  const auto offset_bytes = storage_offset * element_size;
  const auto req_size = offset_bytes + req_pad_bytes;
  TORCH_CHECK(
      req_size <= signal_pad_size,
      "SymmetricMemory::get_signal_pad: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      signal_pad_size,
      " bytes)");
  auto data_ptr =
      reinterpret_cast<uint8_t*>(get_signal_pad_ptrs()[rank]) + offset_bytes;
  auto device = get_device();
  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::for_blob(data_ptr, shape)
      .options(options)
      .target_device(device)
      .make_tensor();
}

} // namespace c10d::symmetric_memory

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
  m.def(
      "nvshmem_broadcast(Tensor(a!) input, int root, str group_name) -> Tensor(a!)");
  m.def("nvshmem_wait_for_signal(Tensor sigpad, int signal, int peer) -> ()");
  m.def(
      "nvshmem_put_with_signal(Tensor(a) tensor, Tensor(a) sigpad, int signal, int peer) -> ()");
  m.def(
      "nvshmem_all_to_all(Tensor input, Tensor(a!) out, str group_name) -> Tensor(a!)");
  m.def(
      "all_to_all_vdev(Tensor input, Tensor(a!) out, Tensor in_splits, Tensor(a!) out_splits_offsets, str group_name) -> ()");
  m.def(
      "all_to_all_vdev_2d(Tensor input, Tensor(a!) out, Tensor in_splits, Tensor(a!) out_splits_offsets, str group_name, int? major_align=None) -> ()");
  m.def(
      "all_to_all_vdev_2d_offset(Tensor input, Tensor(a!) out, Tensor in_splits_offsets, Tensor(a!) out_splits_offsets, str group_name) -> ()");
  m.def(
      "_make_a2a_exchange_plan(Tensor in_splits, Tensor(a!) src_offsets, Tensor(a!) out_splits, Tensor(a!) dst_offsets, str group_name) -> ()");
  m.def(
      "_all_to_all_get(Tensor input, Tensor(a!) out, Tensor src_offsets, Tensor out_splits, Tensor dst_offsets, str group_name, Tensor? b_start, Tensor? b_len, Tensor? b_head) -> ()");
  m.def(
      "_make_a2a_2d_exchange_plan(Tensor in_splits, Tensor(a!) src_offsets, Tensor(a!) out_splits, Tensor(a!) dst_offsets, str group_name, int? major_align=None) -> ()");
  m.def(
      "_all_to_all_v_2d_index_push(Tensor input, Tensor(a!) out, Tensor topk_indices, Tensor occurrences, Tensor dst_offsets, str group_name, Tensor b_start, Tensor b_len, Tensor b_head) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, Meta, m) {
  m.impl("one_shot_all_reduce", one_shot_all_reduce_meta);
  m.impl("one_shot_all_reduce_copy", one_shot_all_reduce_copy_meta);
}

} // namespace

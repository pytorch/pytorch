#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace {

using namespace c10d::symmetric_memory;

static bool is_finalizing_ = false;

class AllocatorMap {
 public:
  static AllocatorMap& get() {
    static AllocatorMap instance;
    return instance;
  }

  void register_allocator(
      c10::DeviceType device_type,
      c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
    map_[device_type] = std::move(allocator);
  }

  c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
      c10::DeviceType device_type) {
    auto it = map_.find(device_type);
    TORCH_CHECK(
        it != map_.end(),
        "SymmetricMemory does not support device type ",
        device_type);
    return it->second;
  }

  ~AllocatorMap() {
    is_finalizing_ = true;
  }

 private:
  AllocatorMap() = default;
  AllocatorMap(const AllocatorMap&) = delete;
  AllocatorMap& operator=(const AllocatorMap&) = delete;

  std::unordered_map<
      c10::DeviceType,
      c10::intrusive_ptr<SymmetricMemoryAllocator>>
      map_;
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
    const std::string& group_name,
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

  const size_t numel =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
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

namespace c10d {
namespace symmetric_memory {

bool is_finalizing() {
  return is_finalizing_;
}

void register_allocator(
    c10::DeviceType device_type,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
  return AllocatorMap::get().register_allocator(
      device_type, std::move(allocator));
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

const GroupInfo& get_group_info(const std::string& group_name) {
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
    const std::string& group_name,
    std::optional<uint64_t> alloc_id) {
  if (alloc_id.has_value()) {
    return empty_strided_p2p_persistent(
        size, stride, dtype, device, group_name, *alloc_id);
  }
  const size_t numel =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
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
    const at::Tensor& tensor) {
  auto allocator = get_allocator(tensor.device().type());
  return allocator->rendezvous(tensor.data_ptr());
}

c10::intrusive_ptr<SymmetricMemory> get_symmetric_memory(
    const at::Tensor& tensor) {
  auto allocator = get_allocator(tensor.device().type());
  TORCH_CHECK(
      allocator->is_rendezvous_completed(tensor.data_ptr()),
      "SymmetricMemory: must invoke rendezvous on a tensor ",
      "before calling get_symmetric_memory on it");
  return allocator->rendezvous(tensor.data_ptr());
}

} // namespace symmetric_memory
} // namespace c10d

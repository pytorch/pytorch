#include "ProcessGroupGloo.hpp"

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/allreduce_ring_chunked.h>
#include <gloo/broadcast_one_to_all.h>
#include <gloo/cuda_allreduce_halving_doubling.h>
#include <gloo/cuda_allreduce_ring_chunked.h>
#include <gloo/cuda_broadcast_one_to_all.h>
#include <gloo/rendezvous/context.h>
#include <gloo/transport/tcp/device.h>

#include <THC.h>

#include <c10d/private/CUDAUtils.hpp>

#define GENERATE_ALL_TYPES(type, func, args...)        \
  switch (type) {                                      \
    case ::at::ScalarType::Float:                      \
      func<float>(args);                               \
      break;                                           \
    case ::at::ScalarType::Double:                     \
      func<double>(args);                              \
      break;                                           \
    case ::at::ScalarType::Half:                       \
      func<gloo::float16>(args);                       \
      break;                                           \
    case ::at::ScalarType::Char:                       \
      func<int8_t>(args);                              \
      break;                                           \
    case ::at::ScalarType::Byte:                       \
      func<uint8_t>(args);                             \
      break;                                           \
    case ::at::ScalarType::Int:                        \
      func<int32_t>(args);                             \
      break;                                           \
    case ::at::ScalarType::Long:                       \
      func<int64_t>(args);                             \
      break;                                           \
    default:                                           \
      throw std::runtime_error("Invalid scalar type"); \
  }

namespace c10d {

using KeyType = AlgorithmKey;
using EntryType = std::unique_ptr<AlgorithmEntry>;

namespace {

// Wrap c10d store as Gloo store
class GlooStore : public ::gloo::rendezvous::Store {
 public:
  GlooStore(const std::shared_ptr<::c10d::Store>& store) : store_(store) {}

  void set(const std::string& key, const std::vector<char>& value) override {
    std::vector<uint8_t> tmp(value.begin(), value.end());
    store_->set(key, tmp);
  }

  std::vector<char> get(const std::string& key) override {
    auto value = store_->get(key);
    return std::vector<char>(value.begin(), value.end());
  }

  void wait(const std::vector<std::string>& keys) override {
    store_->wait(keys, Store::kDefaultTimeout);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    store_->wait(keys, timeout);
  }

 protected:
  std::shared_ptr<::c10d::Store> store_;
};

template <typename T>
const ::gloo::ReductionFunction<T>* reductionFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ::gloo::ReductionFunction<T>::sum;
    case ReduceOp::PRODUCT:
      return ::gloo::ReductionFunction<T>::product;
    case ReduceOp::MIN:
      return ::gloo::ReductionFunction<T>::min;
    case ReduceOp::MAX:
      return ::gloo::ReductionFunction<T>::max;
    case ReduceOp::UNUSED:
      break;
  }

  throw std::runtime_error("Unhandled ReduceOp");
}

std::vector<cudaStream_t> getStreamVector(AlgorithmEntry& entry) {
  std::vector<cudaStream_t> streams(entry.streams.size());
  for (size_t i = 0; i < entry.streams.size(); i++) {
    streams[i] = entry.streams[i].getStream();
  }
  return streams;
}

// synchronizeStreams ensures that the private streams associated with
// an algorithm entry wait for the public streams to complete.
void synchronizeStreams(THCState* thcState, AlgorithmEntry* entry) {
  at::DeviceGuard deviceGuard;
  const auto& key = entry->key;
  for (size_t i = 0; i < key.devices.size(); i++) {
    const auto& device = key.devices[i];
    auto publicStream = THCState_getCurrentStreamOnDevice(thcState, device);
    auto privateStream = entry->streams[i].getStream();
    auto event = entry->events[i].getEvent();

    // Synchronize private stream with public stream.
    //
    // We must use the device guard to cover the case where the public
    // stream is stream 0 and cudaEventRecord relies on the current
    // device to find the right one.
    //
    deviceGuard.set_index(key.devices[i]);
    C10D_CUDA_CHECK(cudaEventRecord(event, publicStream));
    C10D_CUDA_CHECK(cudaStreamWaitEvent(privateStream, event, 0));
  }
}

} // namespace

ProcessGroupGloo::WorkGloo::WorkGloo() : completed_(false), cuda_(false) {}

ProcessGroupGloo::WorkGloo::~WorkGloo() {}

bool ProcessGroupGloo::WorkGloo::isCompleted() const {
  return completed_;
}

bool ProcessGroupGloo::WorkGloo::isSuccess() const {
  return !ex_;
}

void ProcessGroupGloo::WorkGloo::synchronize() {
  if (cuda_) {
    auto thcState = ::at::globalContext().lazyInitCUDA();
    for (size_t i = 0; i < devices_.size(); i++) {
      auto stream = THCState_getCurrentStreamOnDevice(thcState, devices_[i]);
      auto event = events_[i].getEvent();
      C10D_CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
    }
  }
}

bool ProcessGroupGloo::WorkGloo::wait() {
  std::unique_lock<std::mutex> lock(m_);
  while (!completed_) {
    cv_.wait(lock);
  }
  auto success = isSuccess();
  if (success) {
    synchronize();
  }
  return success;
}

const std::exception& ProcessGroupGloo::WorkGloo::exception() const {
  return *ex_;
}

void ProcessGroupGloo::WorkGloo::finish(const AlgorithmEntry& entry) {
  {
    std::unique_lock<std::mutex> lock(m_);
    completed_ = true;
    cuda_ = entry.key.type->is_cuda();

    // Populate devices and events so that we can later synchronize
    // with the operation associated with this work finishing.
    if (cuda_) {
      at::DeviceGuard deviceGuard;
      devices_ = entry.key.devices;
      events_.resize(devices_.size());
      for (size_t i = 0; i < devices_.size(); i++) {
        deviceGuard.set_index(devices_[i]);
        events_[i] = CUDAEvent::create();
        const auto& event = events_[i].getEvent();
        const auto& stream = entry.streams[i].getStream();
        C10D_CUDA_CHECK(cudaEventRecord(event, stream));
      }
    }
  }
  cv_.notify_all();
}

void ProcessGroupGloo::WorkGloo::finishWithException(
    const ::gloo::Exception& ex) {
  {
    std::unique_lock<std::mutex> lock(m_);
    completed_ = true;
    ex_ = std::unique_ptr<::gloo::Exception>(new ::gloo::Exception(ex));
  }
  cv_.notify_all();
}

ProcessGroupGloo::Options::Options()
    : timeout(std::chrono::milliseconds(10 * 1000)),
      threads(2),
      cacheNumAlgorithmEntries(1) {}

ProcessGroupGloo::ProcessGroupGloo(
    const std::shared_ptr<Store>& store,
    int rank,
    int size,
    Options options)
    : ProcessGroup(rank, size),
      store_(new GlooStore(store)),
      stop_(false),
      cacheNumAlgorithmEntries_(options.cacheNumAlgorithmEntries) {
  auto& devices = options.devices;
  if (devices.empty()) {
    throw std::runtime_error("No device(s) specified");
  }

  for (auto& device : options.devices) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
    context->setTimeout(options.timeout);
    context->connectFullMesh(*store_, device);
    contexts_.push_back(std::move(context));
  }

  threads_.resize(options.threads);
  for (size_t i = 0; i < threads_.size(); i++) {
    threads_[i] = std::thread(&ProcessGroupGloo::runLoop, this);
  }

  thcState_ = ::at::globalContext().lazyInitCUDA();
}

ProcessGroupGloo::~ProcessGroupGloo() {
  std::unique_lock<std::mutex> lock(queueMutex_);
  while (!queue_.empty()) {
    queueConsumeCV_.wait(lock);
  }

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  queueProduceCV_.notify_all();
  lock.unlock();

  // Wait for worker threads to terminate
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ProcessGroupGloo::runLoop(void) {
  std::unique_lock<std::mutex> lock(queueMutex_);

  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto tuple = std::move(queue_.front());
    queue_.pop_front();
    queueConsumeCV_.notify_one();

    // Continue holding onto the lock; this ensures that we serialize
    // creation of Gloo algorithm instances for the context associated
    // with this process group.
    auto& entry = std::get<0>(tuple);
    if (!entry->algorithm) {
      createAlgorithm(*entry);
    }

    lock.unlock();
    runSingle(std::move(tuple));
    lock.lock();
  }
}

void ProcessGroupGloo::runSingle(WorkType tuple) {
  auto& entry = std::get<0>(tuple);
  auto& work = std::get<1>(tuple);

  try {
    entry->run();
    work->finish(*entry);
  } catch (const ::gloo::Exception& ex) {
    work->finishWithException(ex);
  }

  // Unblock anyone waiting for this algorithm entry
  std::unique_lock<std::mutex> lock(entry->m);
  entry->busy = false;
  entry->cv.notify_one();
}

void ProcessGroupGloo::createAlgorithm(AlgorithmEntry& entry) {
  const auto& key = entry.key;
  switch (key.collectiveType) {
    case CollectiveType::ALLREDUCE:
      GENERATE_ALL_TYPES(key.type->scalarType(), createAllreduce, entry);
      return;
    case CollectiveType::BROADCAST:
      GENERATE_ALL_TYPES(key.type->scalarType(), createBroadcast, entry);
      return;
    case CollectiveType::UNUSED:
      break;
  }

  throw std::runtime_error("Unhandled collective type");
}

template <typename T>
void ProcessGroupGloo::createAllreduce(AlgorithmEntry& entry) {
  const auto& key = entry.key;
  const auto& backend = key.type->backend();

  // Create algorithm against first context
  auto& context = contexts_[0];
  at::DeviceGuard guard(entry.src[0]);

  if (backend == at::Backend::CPU) {
    if (getSize() < 16) {
      entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::AllreduceRingChunked<T>(
              context,
              getDataPointers<T>(entry.src),
              entry.src[0].numel(),
              reductionFunction<T>(key.reduceOp)));
    } else {
      entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::AllreduceHalvingDoubling<T>(
              context,
              getDataPointers<T>(entry.src),
              entry.src[0].numel(),
              reductionFunction<T>(key.reduceOp)));
    }
    return;
  }

  if (backend == at::Backend::CUDA) {
    if (getSize() < 16) {
      entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::CudaAllreduceRingChunked<T>(
              context,
              getDataPointers<T>(entry.src),
              entry.src[0].numel(),
              getStreamVector(entry)));
    } else {
      entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::CudaAllreduceHalvingDoubling<T>(
              context,
              getDataPointers<T>(entry.src),
              entry.src[0].numel(),
              getStreamVector(entry)));
    }
    return;
  }

  throw std::runtime_error(
      "Unhandled backend: " + std::string(at::toString(backend)));
}

template <typename T>
void ProcessGroupGloo::createBroadcast(AlgorithmEntry& entry) {
  const auto& key = entry.key;
  const auto& backend = key.type->backend();

  // Create algorithm against first context
  auto& context = contexts_[0];
  at::DeviceGuard guard(entry.src[0]);

  if (backend == at::Backend::CPU) {
    entry.algorithm =
        std::unique_ptr<::gloo::Algorithm>(new ::gloo::BroadcastOneToAll<T>(
            context,
            getDataPointers<T>(entry.src),
            entry.src[0].numel(),
            key.srcRank,
            key.srcTensor));
    return;
  }

  if (backend == at::Backend::CUDA) {
    entry.algorithm =
        std::unique_ptr<::gloo::Algorithm>(new ::gloo::CudaBroadcastOneToAll<T>(
            context,
            getDataPointers<T>(entry.src),
            entry.src[0].numel(),
            key.srcRank,
            key.srcTensor,
            getStreamVector(entry)));
    return;
  }

  throw std::runtime_error(
      "Unhandled backend: " + std::string(at::toString(backend)));
}

// Constructs an AlgorithmEntry instance, except for the algorithm
// itself. It allocates the temporary input/output tensors necessary
// to have a fixed address to pass to the Gloo algorithms. The
// AlgorithmEntry is lazily allocated and reused for collective calls
// with the same signature.
//
// Construction of the Gloo algorithm itself it delayed until a thread
// picks up the work, because it performs I/O and can fail. Any I/O
// failure must be signaled through the Work future.
//
EntryType ProcessGroupGloo::construct(const AlgorithmKey& key) {
  at::DeviceGuard deviceGuard;
  auto entry = std::unique_ptr<AlgorithmEntry>(new AlgorithmEntry);
  entry->key = key;

  // Allocate source tensors for this entry
  auto& srcSizes = key.srcSizes;
  entry->src.resize(srcSizes.size());
  for (size_t i = 0; i < srcSizes.size(); i++) {
    deviceGuard.set_index(key.type->is_cuda() ? key.devices[i] : -1);
    entry->src[i] = key.type->tensor(srcSizes[i]);
  }

  // If these are CUDA tensors, create streams and events
  if (key.type->is_cuda()) {
    entry->streams.resize(key.devices.size());
    entry->events.resize(key.devices.size());
    for (size_t i = 0; i < key.devices.size(); i++) {
      deviceGuard.set_index(key.devices[i]);
      entry->streams[i] = CUDAStream::create();
      entry->events[i] = CUDAEvent::create();
    }
  }

  return entry;
}

AlgorithmEntry* ProcessGroupGloo::checkout(const AlgorithmKey& key) {
  auto& vec = cache_[key];
  const auto i = cacheCurrentEntry_[key];

  // Ensure the cache vector is appropriately sized
  if (vec.size() != cacheNumAlgorithmEntries_) {
    vec.resize(cacheNumAlgorithmEntries_);
  }

  // The next call must use the next entry
  cacheCurrentEntry_[key] = (i + 1) % cacheNumAlgorithmEntries_;

  // If there is no entry for this key, create a new one
  if (!vec[i]) {
    vec[i] = construct(key);
  }

  auto& entry = vec[i];

  // Ensure entry is not in use
  std::unique_lock<std::mutex> lock(entry->m);
  while (entry->busy) {
    entry->cv.wait(lock);
  }

  // Mark entry in use
  entry->busy = true;
  return entry.get();
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::enqueue(
    AlgorithmEntry* entry) {
  auto work = std::make_shared<WorkGloo>();
  std::unique_lock<std::mutex> lock(queueMutex_);
  queue_.push_back(std::make_tuple(entry, work));
  queueProduceCV_.notify_one();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  assertSameSizeAndType(tensors);

  AlgorithmKey key;
  key.collectiveType = CollectiveType::BROADCAST;
  key.type = &tensors[0].type();
  key.devices = getDevices(tensors);
  key.srcSizes = getSizes(tensors);
  key.srcRank = opts.rootRank;
  key.srcTensor = opts.rootTensor;

  // Retrieve (create or wait for) pointer to cache entry
  auto entry = checkout(key);

  // Only copy root tensor
  if (getRank() == opts.rootRank) {
    entry->src[opts.rootTensor].copy_(tensors[opts.rootTensor]);
  }

  // In case of CUDA, ensure that operations that are queued after
  // this collective wait for the collective to complete.
  if (key.type->is_cuda()) {
    synchronizeStreams(thcState_, entry);
    entry->run = [=]() mutable {
      entry->algorithm->run();
      for (size_t i = 0; i < tensors.size(); i++) {
        // The THCStreamGuard is a RAII wrapper for temporarily
        // overriding the current THCStream. This also sets the
        // current device to the stream's device.
        THCStreamGuard guard(thcState_, entry->streams[i]);
        tensors[i].copy_(entry->src[i]);
      }
    };
  } else {
    entry->run = [=]() mutable {
      entry->algorithm->run();
      for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].copy_(entry->src[i]);
      }
    };
  }

  return enqueue(entry);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  assertSameSizeAndType(tensors);

  AlgorithmKey key;
  key.collectiveType = CollectiveType::ALLREDUCE;
  key.type = &tensors[0].type();
  key.srcSizes = getSizes(tensors);
  key.devices = getDevices(tensors);
  key.reduceOp = opts.reduceOp;

  // Retrieve (create or wait for) cache entry
  auto entry = checkout(key);

  // Copy input tensors
  for (size_t i = 0; i < tensors.size(); i++) {
    entry->src[i].copy_(tensors[i]);
  }

  // In case of CUDA, ensure that operations that are queued after
  // this collective wait for the collective to complete.
  if (key.type->is_cuda()) {
    synchronizeStreams(thcState_, entry);
    entry->run = [=]() mutable {
      entry->algorithm->run();
      for (size_t i = 0; i < tensors.size(); i++) {
        // The THCStreamGuard is a RAII wrapper for temporarily
        // overriding the current THCStream. This also sets the
        // current device to the stream's device.
        THCStreamGuard guard(thcState_, entry->streams[i]);
        tensors[i].copy_(entry->src[i]);
      }
    };
  } else {
    entry->run = [=]() mutable {
      entry->algorithm->run();
      for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].copy_(entry->src[i]);
      }
    };
  }

  return enqueue(entry);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support reduce");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::allgather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support allgather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int* /* unused */) {
  throw std::runtime_error("ProcessGroupGloo does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::barrier() {
  throw std::runtime_error("ProcessGroupGloo does not support barrier");
}

} // namespace c10d

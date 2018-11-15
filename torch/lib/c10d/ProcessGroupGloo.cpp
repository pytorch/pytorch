#include "ProcessGroupGloo.hpp"

#include <gloo/allgather.h>
#include <gloo/allreduce.h>
#include <gloo/allreduce_halving_doubling.h>
#include <gloo/allreduce_ring_chunked.h>
#include <gloo/barrier_all_to_one.h>
#include <gloo/broadcast.h>
#include <gloo/broadcast_one_to_all.h>
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/scatter.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>

#include <gloo/cuda_allreduce_halving_doubling.h>
#include <gloo/cuda_allreduce_ring_chunked.h>
#include <gloo/cuda_broadcast_one_to_all.h>
#endif

#include <gloo/rendezvous/context.h>
#include <gloo/transport/tcp/device.h>

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

typedef void (*ReduceFunc)(void*, const void*, const void*, size_t);

template <typename T>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::UNUSED:
      break;
  }

  throw std::runtime_error("Unhandled ReduceOp");
}

#ifdef USE_CUDA
std::vector<cudaStream_t> getStreamVector(AlgorithmEntry& entry) {
  std::vector<cudaStream_t> streams;
  streams.reserve(entry.streams.size());
  for (auto s : entry.streams) {
    streams.push_back(s);
  }
  return streams;
}

// synchronizeStreams ensures that the private streams associated with
// an algorithm entry wait for the public streams to complete.
void synchronizeStreams(AlgorithmEntry* entry) {
  const auto& key = entry->key;
  for (size_t i = 0; i < key.devices.size(); i++) {
    const auto& device = key.devices[i];
    auto publicStream = at::cuda::getCurrentCUDAStream(device);
    auto privateStream = entry->streams[i];
    auto& event = entry->events[i];
    // Synchronizes private stream with public stream
    event.record(publicStream);
    event.block(privateStream);
  }
}
#endif

template <typename T, typename O>
void setInputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setOutputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setOutputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

#ifdef USE_CUDA

at::Tensor pinnedLike(at::Tensor& tensor) {
  auto& type = tensor.type().toBackend(at::Backend::CPU);
  auto* allocator = at::cuda::getPinnedMemoryAllocator();
  return type.tensorWithAllocator(tensor.sizes(), tensor.strides(), allocator);
}

// This function initializes a vector of CUDA streams, one for every
// tensor in the input vector, and ensures that these streams are
// synchronized with the current default streams. This is needed so
// that new work on the new streams is serialized w.r.t. all operations
// on the tensors in the input vector.
void initializeStreamsEvents(
    std::vector<at::Tensor>& inputs,
    std::vector<at::cuda::CUDAStream>& streams,
    std::vector<at::cuda::CUDAEvent>& events) {
  at::cuda::OptionalCUDAGuard guard;
  streams.reserve(inputs.size());
  events.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    guard.set_index(inputs[i].get_device());
    // Record event on current stream
    events[i].record(at::cuda::getCurrentCUDAStream());
    // Get a non-default stream to execute asynchronous CUDA operations
    // on for this input. This ensures that the default stream used
    // by the caller is not occupied by c10d related operations.
    streams.push_back(at::cuda::getStreamFromPool(
        /* isHighPriority */ true, inputs[i].get_device()));
    // Ensure the new stream is synchronized with the current stream.
    events[i].block(streams[i]);
  }
}

#endif

} // namespace

bool ProcessGroupGloo::AsyncWork::isCompleted() {
  return completed_;
}

bool ProcessGroupGloo::AsyncWork::isSuccess() const {
  return eptr_ == nullptr;
}

void ProcessGroupGloo::AsyncWork::synchronize() {}

bool ProcessGroupGloo::AsyncWork::wait() {
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

const std::exception& ProcessGroupGloo::AsyncWork::exception() const {
  std::rethrow_exception(eptr_);
}

void ProcessGroupGloo::AsyncWork::finish(std::exception_ptr eptr) {
  std::unique_lock<std::mutex> lock(m_);
  completed_ = true;
  eptr_ = eptr;
  cv_.notify_all();
}

ProcessGroupGloo::WorkGloo::WorkGloo()
    : completed_(false)
#ifdef USE_CUDA
      ,
      cuda_(false)
#endif
{
}

ProcessGroupGloo::WorkGloo::~WorkGloo() {}

bool ProcessGroupGloo::WorkGloo::isCompleted() {
  return completed_;
}

bool ProcessGroupGloo::WorkGloo::isSuccess() const {
  return !ex_;
}

void ProcessGroupGloo::WorkGloo::synchronize() {
#ifdef USE_CUDA
  if (cuda_) {
    for (size_t i = 0; i < devices_.size(); i++) {
      auto stream = at::cuda::getCurrentCUDAStream(devices_[i]);
      events_[i].block(stream);
    }
  }
#endif
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
    if (entry.key.type != nullptr) {
#ifdef USE_CUDA
      cuda_ = entry.key.type->is_cuda();

      // Populate devices and events so that we can later synchronize
      // with the operation associated with this work finishing.
      if (cuda_) {
        devices_ = entry.key.devices;
        events_.resize(devices_.size());
        for (size_t i = 0; i < devices_.size(); i++) {
          const auto& stream = entry.streams[i];
          events_[i].record(stream);
        }
      }
#endif
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

ProcessGroupGloo::SendWork::SendWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer)
    : tensor_(tensor), buffer_(std::move(buffer)) {}

bool ProcessGroupGloo::SendWork::isCompleted() {
  // No way to poll for completion yet
  return true;
}

bool ProcessGroupGloo::SendWork::isSuccess() const {
  // No way to fail yet
  return true;
}

void ProcessGroupGloo::SendWork::synchronize() {
  // CPU only, no need to synchronize
  return;
}

bool ProcessGroupGloo::SendWork::wait() {
  buffer_->waitSend();
  return true;
}

const std::exception& ProcessGroupGloo::SendWork::exception() const {
  throw std::runtime_error("no exception");
}

ProcessGroupGloo::RecvWork::RecvWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    int* srcRank)
    : tensor_(tensor), buffer_(std::move(buffer)), srcRank_(srcRank) {}

bool ProcessGroupGloo::RecvWork::isCompleted() {
  // No way to poll for completion yet
  return true;
}

bool ProcessGroupGloo::RecvWork::isSuccess() const {
  // No way to fail yet
  return true;
}

void ProcessGroupGloo::RecvWork::synchronize() {
  // CPU only, no need to synchronize
  return;
}

bool ProcessGroupGloo::RecvWork::wait() {
  buffer_->waitRecv(srcRank_);
  return true;
}

const std::exception& ProcessGroupGloo::RecvWork::exception() const {
  throw std::runtime_error("no exception");
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
      collectiveCounter_(0),
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

uint32_t ProcessGroupGloo::nextTag() {
  return collectiveCounter_++;
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

    // If we're dealing with only a function, execute it here.
    // This is the case for operations that use the AsyncWork infrastructure
    // and have the work object bound to the function we're calling here.
    if (std::get<0>(tuple) == nullptr) {
      auto& fn = std::get<2>(tuple);
      lock.unlock();
      fn();
      lock.lock();
      continue;
    }

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
    case CollectiveType::BARRIER:
      entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::BarrierAllToOne(contexts_[0]));
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
  at::OptionalDeviceGuard guard(at::device_of(entry.src[0]));

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

#ifdef USE_CUDA
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
#endif

  throw std::runtime_error(
      "Unhandled backend: " + std::string(at::toString(backend)));
}

template <typename T>
void ProcessGroupGloo::createBroadcast(AlgorithmEntry& entry) {
  const auto& key = entry.key;
  const auto& backend = key.type->backend();

  // Create algorithm against first context
  auto& context = contexts_[0];
  at::OptionalDeviceGuard guard(device_of(entry.src[0]));

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

#ifdef USE_CUDA
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
#endif

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
#ifdef USE_CUDA
  at::cuda::OptionalCUDAGuard deviceGuard;
#endif
  auto entry = std::unique_ptr<AlgorithmEntry>(new AlgorithmEntry);
  entry->key = key;

  // Without type there is nothing else to construct
  if (key.type == nullptr) {
    return entry;
  }

  // Allocate source tensors for this entry
  auto& srcSizes = key.srcSizes;
  entry->src.resize(srcSizes.size());
  for (size_t i = 0; i < srcSizes.size(); i++) {
#ifdef USE_CUDA
    deviceGuard.set_index(key.type->is_cuda() ? key.devices[i] : -1);
#else
    if (key.type->is_cuda()) {
      throw std::runtime_error("ProcessGroupGloo is not built with CUDA");
    }
#endif
    entry->src[i] = at::empty(srcSizes[i], key.type->options());
  }

#ifdef USE_CUDA
  // If these are CUDA tensors, create streams and events
  if (key.type->is_cuda()) {
    entry->streams.reserve(key.devices.size());
    entry->events.resize(key.devices.size());
    for (size_t i = 0; i < key.devices.size(); i++) {
      entry->streams.push_back(
          at::cuda::getStreamFromPool(true, key.devices[i]));
    }
  }
#endif

  return entry;
}

AlgorithmEntry* ProcessGroupGloo::checkout(const AlgorithmKey& key) {
  auto& vec = cache_[key];
  const auto i = cacheCurrentEntry_[key];

  // Ensure the cache vector is appropriately sized
  if (vec.size() != static_cast<size_t>(cacheNumAlgorithmEntries_)) {
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
  queue_.push_back(std::make_tuple(entry, work, nullptr));
  queueProduceCV_.notify_one();
  return work;
}

void ProcessGroupGloo::enqueue(std::function<void()> fn) {
  std::unique_lock<std::mutex> lock(queueMutex_);
  queue_.push_back(std::make_tuple(nullptr, nullptr, fn));
  queueProduceCV_.notify_one();
}

namespace {

class AsyncBroadcastWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncBroadcastWork(
      std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag)
      : context(context),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor& tensor) {
    const auto& scalarType = tensor.scalar_type();
    gloo::BroadcastOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::broadcast(opts);
  }

  void run() override {
    broadcast(inputs[rootTensor]);

    // Copy to non-root tensors
    for (size_t i = 0; i < inputs.size(); i++) {
      if (i == rootTensor) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
};

#ifdef USE_CUDA

class AsyncBroadcastCUDAWork : public AsyncBroadcastWork {
 public:
  AsyncBroadcastCUDAWork(
      std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag)
      : AsyncBroadcastWork(context, inputs, rootRank, rootTensor, tag) {
    initializeStreamsEvents(inputs, streams, events);

    // Create pinned host side tensors.
    tmp = pinnedLike(inputs[rootTensor]);
    at::cuda::OptionalCUDAStreamGuard guard;
    if (context->rank == rootRank) {
      guard.reset_stream(streams[rootTensor]);
      tmp.copy_(inputs[rootTensor], /* non_blocking */ true);
    }
  }

  void run() override {
    at::cuda::OptionalCUDAStreamGuard guard;

    // Synchronize with copy operation if applicable.
    if (context->rank == rootRank) {
      guard.reset_stream(streams[rootTensor]);
      AT_CUDA_CHECK(cudaStreamSynchronize(streams[rootTensor]));
    }

    // Run broadcast on host side tensors.
    broadcast(tmp);

    // Kick off copy back to the CUDA tensors.
    for (size_t i = 0; i < inputs.size(); i++) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp, /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    at::cuda::OptionalCUDAGuard guard;

    // Synchronize with the copy back to CUDA tensors.
    for (size_t i = 0; i < inputs.size(); i++) {
      guard.set_index(inputs[i].get_device());
      events[i].block(at::cuda::getCurrentCUDAStream());
    }
  }

  at::Tensor tmp;
  std::vector<at::cuda::CUDAStream> streams;
  std::vector<at::cuda::CUDAEvent> events;
};

#endif

} // namespace

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::broadcast(
    std::vector<at::Tensor>& inputs,
    const BroadcastOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupGloo::broadcast: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, inputs.size());
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
#ifdef USE_CUDA
    case at::kCUDA:
#endif
      break;
    default:
      invalidArgument("unsupported device type");
  }

  std::shared_ptr<AsyncBroadcastWork> work;
  auto& context = contexts_[0];
  if (device.type() == at::kCPU) {
    work = std::make_shared<AsyncBroadcastWork>(
        context, inputs, opts.rootRank, opts.rootTensor, nextTag());
#ifdef USE_CUDA
  } else if (device.type() == at::kCUDA) {
    work = std::make_shared<AsyncBroadcastCUDAWork>(
        context, inputs, opts.rootRank, opts.rootTensor, nextTag());
#endif
  } else {
    throw std::runtime_error("Invalid backend");
  }

  enqueue(std::bind(AsyncWork::execute, work));
  return work;
}

namespace {

class AsyncAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllreduceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag)
      : context(context), inputs(inputs), reduceOp(reduceOp), tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();
    gloo::AllreduceOptions opts(context);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors);
    gloo::allreduce(opts);
  }

  void run() override {
    allreduce(inputs);
  }

  template <typename T>
  void getFunction(gloo::AllreduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::AllreduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp op) {
    gloo::AllreduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

#ifdef USE_CUDA

class AsyncAllreduceCUDAWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag)
      : AsyncAllreduceWork(context, inputs, reduceOp, tag) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    at::cuda::OptionalCUDAStreamGuard guard;
    for (size_t i = 0; i < inputs.size(); i++) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    at::cuda::OptionalCUDAGuard device_guard;
    for (size_t i = 0; i < inputs.size(); i++) {
      device_guard.set_index(inputs[i].get_device());
      AT_CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Run allreduce on host side tensors.
    allreduce(tmp);

    // Kick off copy back to the CUDA tensors.
    at::cuda::OptionalCUDAStreamGuard stream_guard;
    for (size_t i = 0; i < inputs.size(); i++) {
      stream_guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    at::cuda::OptionalCUDAGuard guard;
    for (size_t i = 0; i < inputs.size(); i++) {
      guard.set_index(static_cast<at::DeviceIndex>(inputs[i].get_device()));
      events[i].block(at::cuda::getCurrentCUDAStream());
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<at::cuda::CUDAStream> streams;
  std::vector<at::cuda::CUDAEvent> events;
};

#endif

} // namespace

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::allreduce(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupGloo::allreduce: " + msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
#ifdef USE_CUDA
    case at::kCUDA:
#endif
      break;
    default:
      invalidArgument("unsupported device type");
  }

  std::shared_ptr<AsyncAllreduceWork> work;
  auto& context = contexts_[0];
  if (device.type() == at::kCPU) {
    work = std::make_shared<AsyncAllreduceWork>(
        context, inputs, opts.reduceOp, nextTag());
#ifdef USE_CUDA
  } else if (device.type() == at::kCUDA) {
    work = std::make_shared<AsyncAllreduceCUDAWork>(
        context, inputs, opts.reduceOp, nextTag());
#endif
  } else {
    throw std::runtime_error("Invalid backend");
  }

  enqueue(std::bind(AsyncWork::execute, work));
  return work;
}

namespace {

class AsyncReduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncReduceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag)
      : context(context),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        reduceOp(reduceOp),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void run() override {
    const auto& scalarType = inputs[0].scalar_type();
    gloo::ReduceOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, inputs[0]);
    gloo::reduce(opts);
  }

 protected:
  template <typename T>
  void getFunction(gloo::ReduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::ReduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp op) {
    gloo::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

} // namespace

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::reduce(
    std::vector<at::Tensor>& inputs,
    const ReduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupGloo::reduce: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, inputs.size());
  assertSingleElement(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);
  assertCPU(invalidArgument, inputs);

  auto work = std::make_shared<AsyncReduceWork>(
      contexts_[0],
      inputs,
      opts.rootRank,
      opts.rootTensor,
      opts.reduceOp,
      nextTag());
  enqueue(std::bind(AsyncWork::execute, work));
  return work;
}

namespace {

class AsyncAllgatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllgatherWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag)
      : context(context), outputs(outputs), inputs(inputs), tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;

  void run() override {
    const auto& scalarType = inputs[0].scalar_type();
    gloo::AllgatherOptions opts(context);
    opts.setTag(tag);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Use single flat output tensor.
    // The first dimension corresponds to the index into outputs[N],
    // so copying into the actual output later is easy.
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    // Unflatten into output tensors.
    for (size_t i = 0; i < outputs.size(); i++) {
      for (size_t j = 0; j < outputs[i].size(); j++) {
        outputs[i][j].copy_(flatOutputTensor[j]);
      }
    }
  }
};

} // namespace

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupGloo::allgather: " + msg);
  };

  if (inputs.size() == 0) {
    invalidArgument("requires non-empty input tensor list");
  }

  if (inputs.size() != outputs.size()) {
    invalidArgument(
        "requires input/output tensor lists to have the same length");
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    const auto expected = inputs.size() * getSize();
    const auto actual = outputs[i].size();
    if (actual != expected) {
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) +
          " (expected length " + std::to_string(expected) + ", got " +
          std::to_string(actual) + ")");
    }
  }

  assertDense(invalidArgument, inputs);
  assertCPU(invalidArgument, inputs);

  // Expect all input/output tensors to have the same type and sizes
  const auto& type = inputs[0].type();
  const auto& sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, type, sizes);
  for (size_t i = 0; i < outputs.size(); i++) {
    assertTypeAndSizesMatch(invalidArgument, outputs[i], type, sizes);
  }

  auto work = std::make_shared<AsyncAllgatherWork>(
      contexts_[0], outputs, inputs, nextTag());
  enqueue(std::bind(AsyncWork::execute, work));
  return work;
}

namespace {

class AsyncGatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncGatherWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      int root,
      uint32_t tag)
      : context(context),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const int root;
  const uint32_t tag;

  void run() override {
    const auto scalarType = inputs[0].type().scalarType();
    gloo::GatherOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set single temporary tensor on root process.
    // This is later scattered to the separate output tensors.
    at::Tensor flatOutputTensor;
    if (context->rank == root) {
      flatOutputTensor = newLikeFlat(outputs[0]);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    }

    // Set single input tensor on all processes.
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
    gloo::gather(opts);

    // Unflatten into output tensors on root process.
    if (context->rank == root) {
      for (size_t i = 0; i < outputs[0].size(); i++) {
        outputs[0][i].copy_(flatOutputTensor[i]);
      }
    }
  }
};

} // namespace

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupGloo::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementInput(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);
  assertCPU(invalidArgument, inputs);

  if (getRank() == opts.rootRank) {
    if (outputs.size() != 1 || outputs[0].size() != getSize()) {
      invalidArgument(
          "requires a single-element output list "
          "containing a list with <size> tensors");
    }

    const auto& type = inputs[0].type();
    const auto& sizes = inputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputs[0], type, sizes);
  } else {
    if (outputs.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
  }

  auto work = std::make_shared<AsyncGatherWork>(
      contexts_[0], outputs, inputs, opts.rootRank, nextTag());
  enqueue(std::bind(AsyncWork::execute, work));
  return work;
}

namespace {

class AsyncScatterWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncScatterWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag)
      : context(context),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> outputs;
  std::vector<std::vector<at::Tensor>> inputs;
  const int root;
  const uint32_t tag;

  void run() override {
    const auto scalarType = outputs[0].type().scalarType();
    gloo::ScatterOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set list of input tensors on root process
    if (context->rank == root) {
      GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
    }

    // Set single output tensor on all processes
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
    gloo::scatter(opts);
  }
};

} // namespace

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupGloo::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementOutput(invalidArgument, outputs);
  assertDense(invalidArgument, outputs);
  assertCPU(invalidArgument, outputs);

  if (getRank() == opts.rootRank) {
    if (inputs.size() != 1 || inputs[0].size() != getSize()) {
      invalidArgument(
          "requires a single-element input list "
          "containing a list with <size> tensors");
    }
    const auto& type = outputs[0].type();
    const auto& sizes = outputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputs[0], type, sizes);
  } else {
    if (inputs.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
  }

  auto work = std::make_shared<AsyncScatterWork>(
      contexts_[0], outputs, inputs, opts.rootRank, nextTag());
  enqueue(std::bind(AsyncWork::execute, work));
  return work;
}

at::Tensor& checkSingleTensor(std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error("ProcessGroupGloo::send takes a single tensor");
  }
  auto& tensor = tensors[0];
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    throw std::runtime_error("input tensor has to be dense");
  }
  return tensor;
}

uint32_t checkTag(int32_t tag) {
  if (tag < 0) {
    throw std::runtime_error("Tag must be >= 0");
  }
  return (uint32_t)tag;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.data_ptr();
  auto size = tensor.numel() * tensor.type().elementSizeInBytes();

  // Construct unbound buffer.
  auto& context = contexts_[0];
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->send(dstRank, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the send.
  return std::make_shared<SendWork>(tensor, std::move(buf));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.data_ptr();
  auto size = tensor.numel() * tensor.type().elementSizeInBytes();

  // Construct unbound buffer.
  auto& context = contexts_[0];
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->recv(srcRank, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  return std::make_shared<RecvWork>(tensor, std::move(buf), nullptr);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int* srcRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.data_ptr();
  auto size = tensor.numel() * tensor.type().elementSizeInBytes();

  // Construct unbound buffer.
  auto& context = contexts_[0];
  auto buf = context->createUnboundBuffer(ptr, size);

  // Build list of ranks that this operation can recv from. In these
  // bindings we don't differentiate between ranks and can receive
  // from any other process in the group.
  std::vector<int> srcRanks;
  srcRanks.resize(size_);
  for (auto i = 0; i < size_; i++) {
    srcRanks.push_back(i);
  }

  buf->recv(srcRanks, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  return std::make_shared<RecvWork>(tensor, std::move(buf), srcRank);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::barrier() {
  AlgorithmKey key;
  key.collectiveType = CollectiveType::BARRIER;

  auto entry = checkout(key);
  entry->run = [=]() mutable { entry->algorithm->run(); };
  return enqueue(entry);
}

std::unordered_map<int, int> ProcessGroupGloo::getGroupRank() {
  throw std::runtime_error("ProcessGroupGloo does not support getGroupRank");
}

} // namespace c10d

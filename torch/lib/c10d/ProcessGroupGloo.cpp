#include "ProcessGroupGloo.hpp"

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/broadcast_one_to_all.h>
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
  }

  throw std::runtime_error("Unhandled ReduceOp");
}

} // namespace

ProcessGroupGloo::WorkGloo::WorkGloo() : completed_(false) {}

ProcessGroupGloo::WorkGloo::~WorkGloo() {}

bool ProcessGroupGloo::WorkGloo::isCompleted() const {
  return completed_;
}

bool ProcessGroupGloo::WorkGloo::isSuccess() const {
  return !ex_;
}

bool ProcessGroupGloo::WorkGloo::wait() {
  std::unique_lock<std::mutex> lock(m_);
  while (!completed_) {
    cv_.wait(lock);
  }
  return isSuccess();
}

const std::exception& ProcessGroupGloo::WorkGloo::exception() const {
  return *ex_;
}

void ProcessGroupGloo::WorkGloo::finish() {
  {
    std::unique_lock<std::mutex> lock(m_);
    completed_ = true;
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
    : timeout(std::chrono::milliseconds(10 * 1000)), threads(2) {}

ProcessGroupGloo::ProcessGroupGloo(
    const std::shared_ptr<Store>& store,
    int rank,
    int size,
    Options options)
    : ProcessGroup(rank, size), store_(new GlooStore(store)), stop_(false) {
  auto& devices = options.devices;
  if (devices.empty()) {
    devices.push_back(::gloo::transport::tcp::CreateDevice("localhost"));
  }

  for (auto& device : options.devices) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
    context->setTimeout(options.timeout);
    context->connectFullMesh(*store_, device);
    contexts_.push_back(std::move(context));
  }

  threads_.resize(options.threads);
  for (int i = 0; i < threads_.size(); i++) {
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
    work->finish();
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
  }

  throw std::runtime_error("Unhandled collective type");
}

template <typename T>
void ProcessGroupGloo::createAllreduce(AlgorithmEntry& entry) {
  const auto& key = entry.key;

  // Create algorithm against first context
  auto& context = contexts_[0];
  entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::AllreduceHalvingDoubling<T>(
          context,
          getDataPointers<T>(entry.src),
          entry.src[0].numel(),
          reductionFunction<T>(key.reduceOp)));
}

template <typename T>
void ProcessGroupGloo::createBroadcast(AlgorithmEntry& entry) {
  const auto& key = entry.key;

  // Create algorithm against first context
  auto& context = contexts_[0];
  entry.algorithm =
      std::unique_ptr<::gloo::Algorithm>(new ::gloo::BroadcastOneToAll<T>(
          context,
          getDataPointers<T>(entry.src),
          entry.src[0].numel(),
          key.srcRank,
          key.srcTensor));
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
  auto entry = std::unique_ptr<AlgorithmEntry>(new AlgorithmEntry);
  entry->key = key;

  // Allocate source tensors for this entry
  auto& srcSizes = key.srcSizes;
  entry->src.resize(srcSizes.size());
  for (int i = 0; i < srcSizes.size(); i++) {
    entry->src[i] = key.type->tensor(srcSizes[i]);
  }

  return entry;
}

AlgorithmEntry* ProcessGroupGloo::checkout(const AlgorithmKey& key) {
  auto it = cache_.find(key);

  // If there is no entry for this key yet, it must be the first time
  // we see and can create a new entry. Use hard limit of 1 instance
  // per key until we add support for a dynamic limit.
  if (it == cache_.end()) {
    cache_[key] = construct(key);
    it = cache_.find(key);
  }

  auto& entry = it->second;

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

  // Define how to run the algorithm and copy back results
  entry->run = [=]() mutable {
    entry->algorithm->run();
    for (int i = 0; i < tensors.size(); i++) {
      tensors[i].copy_(entry->src[i]);
    }
  };

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
  for (int i = 0; i < tensors.size(); i++) {
    entry->src[i].copy_(tensors[i]);
  }

  // Define how to run the algorithm and copy back results
  entry->run = [=]() mutable {
    entry->algorithm->run();
    for (int i = 0; i < tensors.size(); i++) {
      tensors[i].copy_(entry->src[i]);
    }
  };

  return enqueue(entry);
}

} // namespace c10d

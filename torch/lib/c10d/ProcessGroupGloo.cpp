#include "ProcessGroupGloo.hpp"

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/rendezvous/context.h>

#define GENERATE_ALL_TYPES(type, func, args...)                         \
  switch (type) {                                                       \
    case ::at::ScalarType::Float: func<float>(args); break;             \
    case ::at::ScalarType::Double: func<double>(args); break;           \
    case ::at::ScalarType::Half: func<gloo::float16>(args); break;      \
    case ::at::ScalarType::Char: func<int8_t>(args); break;             \
    case ::at::ScalarType::Byte: func<uint8_t>(args); break;            \
    case ::at::ScalarType::Int: func<int32_t>(args); break;             \
    case ::at::ScalarType::Long: func<int64_t>(args); break;            \
    default:                                                            \
      throw std::runtime_error("Invalid scalar type");                  \
  }

namespace c10d {

using KeyType = AlgorithmKey;
using EntryType = std::unique_ptr<AlgorithmEntry>;

namespace {

// Wrap c10d store as Gloo store
class GlooStore : public ::gloo::rendezvous::Store {
 public:
  GlooStore(const std::shared_ptr<::c10d::Store>& store)
      : store_(store) {
  }

  void set(const std::string& key, const std::vector<char>& value) override {
    // @pietern was unable to figure out a way to cast the value of
    // type std::vector<char> to an std::vector<unsigned char>,
    // so going with a copy here (it's not a hot path so it's fine).
    std::vector<uint8_t> tmp(value.size());
    memcpy(tmp.data(), value.data(), value.size());
    store_->set(key, tmp);
  }

  std::vector<char> get(const std::string& key) override {
    // @pietern was unable to figure out a way to cast the value of
    // type std::vector<unsigned char> to an std::vector<char>,
    // so going with a copy here (it's not a hot path so it's fine).
    auto value = store_->get(key);
    std::vector<char> tmp(value.size());
    memcpy(tmp.data(), value.data(), value.size());
    return tmp;
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

ProcessGroupGloo::WorkGloo::WorkGloo() {
  completed_ = false;
}

ProcessGroupGloo::WorkGloo::~WorkGloo() {
}

bool ProcessGroupGloo::WorkGloo::isCompleted() {
  std::unique_lock<std::mutex> lock(m_);
  return completed_;
}

bool ProcessGroupGloo::WorkGloo::wait() {
  std::unique_lock<std::mutex> lock(m_);
  while (!completed_) {
    cv_.wait(lock);
  }
  return !ex_;
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

ProcessGroupGloo::ProcessGroupGloo(
    const std::shared_ptr<Store>& store,
    const std::vector<std::shared_ptr<::gloo::transport::Device>>& devices,
    int rank,
    int size)
    : ProcessGroup(rank, size),
      store_(new GlooStore(store)),
      devices_(devices) {
  stop_ = false;
}

ProcessGroupGloo::~ProcessGroupGloo() {
  // Require this process group to be explicitly shut down prior to being
  // destructed. This is the easiest way to guarantee clean shutdown
  // and avoid blocking/throwing in a destructor.
}

void ProcessGroupGloo::initialize() {
  contexts_.clear();
  for (auto& device : devices_) {
    auto context = std::shared_ptr<::gloo::rendezvous::Context>(
        new ::gloo::rendezvous::Context(rank_, size_));
    context->setTimeout(std::chrono::milliseconds(100));
    context->connectFullMesh(*store_, device);
    contexts_.push_back(std::move(context));
  }

  // TODO(@pietern): Make number of threads configurable
  threads_.resize(2);
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i] = std::thread(&ProcessGroupGloo::runLoop, this);
  }
}

void ProcessGroupGloo::destroy() {
  std::unique_lock<std::mutex> lock(m_);
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
  std::unique_lock<std::mutex> lock(m_);

  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto tuple = std::move(queue_.front());
    queue_.pop_front();
    queueConsumeCV_.notify_one();

    auto& entry = std::get<0>(tuple);
    auto& work = std::get<1>(tuple);

    // Continue holding onto the lock; this ensures that we serialize
    // creation of Gloo algorithm instances for the context associated
    // with this process group.
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
    entry->run(entry);
    work->finish();
  } catch (const ::gloo::Exception& ex) {
    work->finishWithException(ex);
  }

  // Return entry to cache
  std::unique_lock<std::mutex> lock(m_);
  cache_[entry->key] = std::move(entry);
  cacheCV_.notify_all();
}

void ProcessGroupGloo::createAlgorithm(AlgorithmEntry& entry) {
  const auto& key = entry.key;
  switch (key.collectiveType) {
    case CollectiveType::ALLREDUCE:
      GENERATE_ALL_TYPES(
          key.type->scalarType(),
          createAllreduce,
          entry);
      return;
  }

  throw std::runtime_error("Unhandled collective type");
}

template <typename T>
void ProcessGroupGloo::createAllreduce(AlgorithmEntry& entry) {
  const auto& key = entry.key;

  // Turn source tensors into T*
  auto& src = entry.src;
  std::vector<T*> tmp(src.size());
  for (int i = 0; i < src.size(); i++) {
    tmp[i] = static_cast<T*>(src[i].storage()->data());
  }

  // Create algorithm against first context
  auto& context = contexts_[0];
  entry.algorithm = std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::AllreduceHalvingDoubling<T>(
          context,
          tmp,
          src[0].numel(),
          reductionFunction<T>(key.reduceOp)));
}

// Constructs an AlgorithmEntry instance, except for the Gloo
// algorithm itself. It allocates temporary input/output tensors, CUDA
// streams (if applicable), etcetera. These are allocated lazily for
// every unique collective call and reused for identical collective
// calls. They cannot be allocated asynchronously because our
// collective functions may need thme before queueing asynchronous
// work. For example, to work with asynchronous CUDA code, the
// collective call needs to issue an asynchronous memory copy, and a
// call to cudaStreamWaitEvent to make it wait for the asynchronous
// execution of the collective call to complete.
//
// Construction of the Gloo algorithm itself it delayed until a thread
// picks up the work, because it performs I/O and can fail. Any I/O
// failure must be signaled through the Work future.
//
EntryType ProcessGroupGloo::construct(
    const AlgorithmKey& key) {
  auto entry = std::unique_ptr<AlgorithmEntry>(new AlgorithmEntry);
  entry->key = key;

  // Allocate source tensors for this entry
  auto& srcSizes = key.srcSizes;
  entry->src.resize(srcSizes.size());
  for (int i = 0; i < srcSizes.size(); i++) {
    entry->src[i] = at::zeros(*key.type, at::IntList(srcSizes[i]));
  }

  // TODO(@pietern): Create CUDA streams and events
  return std::move(entry);
}

EntryType ProcessGroupGloo::checkout(const AlgorithmKey& key) {
  std::unique_lock<std::mutex> lock(m_);

  // Initialize number of entries for this key
  if (cacheCreated_.count(key) == 0) {
    cacheCreated_.emplace(key, 0);
  }

  // If there is no entry for this key yet, it must be the first time
  // we see and can create a new entry. Use hard limit of 1 instance
  // per key until we add support for a dynamic limit.
  if (cacheCreated_[key] < 1) {
    cacheCreated_[key]++;
    return std::move(construct(key));
  }

  // Optionally wait for entry to be returned to the cache.
  auto it = cache_.find(key);
  while (it == cache_.end()) {
    cacheCV_.wait(lock);
    it = cache_.find(key);
  }

  auto entry = std::move(it->second);
  cache_.erase(key);
  return std::move(entry);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return std::shared_ptr<Work>(new WorkGloo());
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupGloo::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // Ensure we have at least one tensor
  if (tensors.size() == 0) {
    throw std::invalid_argument("allreduce: argument is empty");
  }

  // Ensure all tensors have identical type and shape
  auto& type = tensors[0].type();
  auto sizes = tensors[0].sizes();
  for (int i = 1; i < tensors.size(); i++) {
    if (tensors[i].type() != type) {
      throw std::invalid_argument("allreduce: argument contains mixed types");
    }
    if (!tensors[i].sizes().equals(sizes)) {
      throw std::invalid_argument("allreduce: argument contains mixed sizes");
    }
  }

  // List of devices for tensors determines uniqueness of this call
  std::vector<int> devices;
  if (type.is_cuda()) {
    devices.resize(tensors.size());
    for (int i = 0; i < tensors.size(); i++) {
      devices[i] = tensors[i].storage()->getDevice();
    }
  }

  AlgorithmKey key;
  key.collectiveType = CollectiveType::ALLREDUCE;
  key.type = &tensors[0].type();
  key.srcSizes.resize(tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    key.srcSizes[i] = tensors[i].sizes();
  }
  key.devices = std::move(devices);
  key.reduceOp = opts.reduceOp;

  // Retrieve (create or wait for) cache entry
  auto entry = checkout(key);

  // Copy input tensors
  for (int i = 0; i < tensors.size(); i++) {
    entry->src[i].copy_(tensors[i]);
  }

  // Define how to run the algorithm and copy back results
  entry->run = [tensors](EntryType& entry) mutable {
    entry->algorithm->run();
    for (int i = 0; i < tensors.size(); i++) {
      tensors[i].copy_(entry->src[i]);
    }
    // TODO(@pietern): cudaEventRecord to unblock source stream
  };

  auto work = std::shared_ptr<WorkGloo>(new WorkGloo());
  std::unique_lock<std::mutex> lock(m_);
  queue_.push_back(std::make_tuple(std::move(entry), work));
  queueProduceCV_.notify_one();

  // TODO(@pietern): cudaStreamWaitEvent on source stream
  return std::move(work);
}

} // namespace c10d

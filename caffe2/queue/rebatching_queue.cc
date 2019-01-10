#include "rebatching_queue.h"
#include "caffe2/utils/smart_tensor_printer.h"

namespace caffe2 {

namespace {

// This concat function will always create a new first dimension to concat
void concat(
    CPUContext& context,
    const std::vector<std::vector<TensorCPU>>& inputs,
    const std::vector<TensorCPU*>& outputs) {
  CAFFE_ENFORCE(!inputs.empty());

  const auto& inputZero = inputs[0];
  const auto numTensors = inputZero.size();
  const auto numRows = inputs.size();

  // Precompute the output sizes to avoid resizing
  std::vector<std::vector<int64_t>> outputDims(numTensors);

  for (int i = 0; i < numTensors; ++i) {
    SmartTensorPrinter::PrintTensor(inputZero.at(i));
    outputDims[i] = inputZero.at(i).dims().vec();
    outputDims[i].insert(outputDims[i].begin(), numRows);
  }

  // Resize to the final output size
  std::vector<void*> destinations(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    outputs[i]->Resize(outputDims[i]);
    destinations[i] = outputs[i]->raw_mutable_data(inputZero[i].meta());
  }

  for (int i = 0; i < numRows; ++i) {
    CAFFE_ENFORCE_EQ(inputs[i].size(), numTensors);

    for (int j = 0; j < numTensors; ++j) {
      const auto& input = inputs[i][j];

      CAFFE_ENFORCE(inputZero[j].meta() == input.dtype());
      CAFFE_ENFORCE_EQ(inputZero[j].itemsize(), input.itemsize());
      CAFFE_ENFORCE_EQ(inputZero[j].ndim(), input.dim());
      for (int k = 0; k < input.dim(); ++k) {
        CAFFE_ENFORCE_EQ(input.sizes()[k], inputZero[j].dims()[k]);
      }

      // Skip empty tensors
      if (input.numel() == 0) {
        continue;
      }

      context.CopyItemsToCPU(
          input.dtype(),
          input.numel(),
          input.raw_data() /* src */,
          destinations[j] /* dst */
      );

      destinations[j] =
          (char*)destinations[j] + input.numel() * input.itemsize();
    }
  }
}

std::vector<std::vector<TensorCPU>> split(
    CPUContext& context,
    const std::vector<const TensorCPU*>& inputs) {
  CAFFE_ENFORCE(!inputs.empty());

  const auto outputSize = inputs[0]->sizes().at(0);
  std::vector<std::vector<TensorCPU>> outputs(outputSize);

  for (const auto* inputPtr : inputs) {
    CAFFE_ENFORCE(inputPtr);

    const auto& input = *inputPtr;
    const auto innerSize = input.size_from_dim(1);
    const auto itemSize = input.dtype().itemsize();

    auto outputDims = input.sizes().vec();
    CAFFE_ENFORCE(!outputDims.empty());
    outputDims.erase(outputDims.begin());
    CAFFE_ENFORCE_EQ(input.sizes().at(0), outputSize);

    for (int i = 0; i < outputSize; ++i) {
      outputs[i].push_back(Tensor(outputDims, CPU));
      context.CopyItemsToCPU(
          input.dtype(),
          innerSize,
          (char*)input.raw_data() + i * innerSize * itemSize /* src */,
          outputs[i].back().raw_mutable_data(input.dtype()) /* dst */);
    }
  }

  return outputs;
}
} // anonymous namespace

RebatchingQueue::RebatchingQueue(size_t capacity, size_t numBlobs)
    : capacity_(capacity), numBlobs_(numBlobs), queue_(capacity) {}

RebatchingQueue::~RebatchingQueue() {
  close();
}

bool RebatchingQueue::canRead() const {
  return tail_ < head_;
}

bool RebatchingQueue::dequeue(
    CPUContext& context,
    size_t numElements,
    const std::vector<TensorCPU*>& outputs) {
  std::vector<std::vector<TensorCPU>> results;
  results.reserve(numElements);

  for (;;) {
    if (results.size() == numElements) {
      break;
    }

    {
      std::unique_lock<std::mutex> lock(mutex_);

      cvEmpty_.wait(lock, [this] { return canRead() || isClosed_; });

      // We only want to stop reading if the queue is empty and closed
      if (!canRead() && isClosed_) {
        break;
      }

      do {
        results.push_back(std::move(queue_[tail_++ % capacity()]));
      } while (canRead() && results.size() < numElements);
    }

    if (numElements == 1) {
      cvOverflow_.notify_one();
    } else {
      cvOverflow_.notify_all();
    }
  }

  if (results.empty()) {
    return false;
  }

  concat(context, results, outputs);

  return true;
}

bool RebatchingQueue::canWrite() const {
  return tail_ + capacity() > head_;
}

bool RebatchingQueue::enqueueOne(
    CPUContext& /*context*/,
    const std::vector<const TensorCPU*>& inputs) {
  std::vector<std::vector<TensorCPU>> splittedInputs;
  splittedInputs.emplace_back();
  auto& tensorVector = splittedInputs.back();
  tensorVector.reserve(inputs.size());
  for (const auto* tensorPtr : inputs) {
    tensorVector.push_back(tensorPtr->Clone());
  }

  return enqueue(std::move(splittedInputs));
}

bool RebatchingQueue::enqueueMany(
    CPUContext& context,
    const std::vector<const TensorCPU*>& inputs) {
  CAFFE_ENFORCE_EQ(numBlobs_, inputs.size());

  std::vector<std::vector<TensorCPU>> splittedInputs;
  splittedInputs = split(context, inputs);
  return enqueue(std::move(splittedInputs));
}

bool RebatchingQueue::enqueue(
    std::vector<std::vector<TensorCPU>> splittedInputs) {
  int idx = 0;
  for (;;) {
    if (idx >= splittedInputs.size()) {
      break;
    }

    {
      std::unique_lock<std::mutex> lock(mutex_);

      cvOverflow_.wait(lock, [this] { return canWrite() || isClosed_; });

      if (isClosed_) {
        // If we are here it means that we didn't apply the entire batch and if
        // we get closed in the middle of enquing we treat it as a non-success.
        return false;
      }

      do {
        queue_[head_++ % capacity()] = std::move(splittedInputs[idx++]);
      } while (canWrite() && idx < splittedInputs.size());
    }

    cvEmpty_.notify_all();
  }

  return true;
}

size_t RebatchingQueue::capacity() const {
  return capacity_;
}

size_t RebatchingQueue::numBlobs() const {
  return numBlobs_;
}

bool RebatchingQueue::isClosed() const {
  std::lock_guard<std::mutex> g(mutex_);
  return isClosed_;
}

void RebatchingQueue::close() {
  {
    std::lock_guard<std::mutex> g(mutex_);
    isClosed_ = true;
  }

  cvEmpty_.notify_all();
  cvOverflow_.notify_all();
}
} // caffe2

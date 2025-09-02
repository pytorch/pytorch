#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGlooDetail.hpp>

#include <gloo/cuda_allreduce_ring_chunked.h>

namespace c10d {

class AsyncAllreduceCUDADeviceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllreduceCUDADeviceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            {inputs},
            OpType::ALLREDUCE,
            seq,
            timeout,
            "gloo:all_reduce",
            inputs),
        inputs_(inputs),
        reduceOp_(reduceOp) {}

  template <typename T>
  void createAlgorithm(std::unique_ptr<gloo::Algorithm>& algo) {
    auto count = inputs_.at(0).numel();
    std::vector<T*> ptrs;
    for (const auto& tensor : inputs_) {
      TORCH_CHECK_EQ(tensor.numel(), count);
      ptrs.push_back(static_cast<T*>(tensor.data_ptr()));
    }
    algo = std::make_unique<
        gloo::CudaAllreduceRingChunked<T, gloo::CudaDeviceWorkspace<T>>>(
        context_, ptrs, count);
  }

  void run() override {
    const auto& scalarType = inputs_.at(0).scalar_type();

    std::unique_ptr<gloo::Algorithm> algo;
    GENERATE_ALL_TYPES(scalarType, createAlgorithm, algo);
    algo->run();

    // Gloo doesn't support AVG so we use SUM + division.
    if (reduceOp_ == ReduceOp::AVG) {
      inputs_[0] /= context_->size;
    } else {
      TORCH_CHECK_EQ(reduceOp_, ReduceOp::SUM);
    }
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs_;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return inputs_;
  }

  void synchronize() override {
    // TODO: is synchronization needed?
  }

 private:
  std::vector<at::Tensor> inputs_;
  const ReduceOp reduceOp_;
};

class AsyncAllreduceCUDAHostWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCUDAHostWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncAllreduceWork(
            context,
            inputs,
            std::move(reduceOp),
            tag,
            seq,
            timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    allreduce(tmp);

    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams{};
  std::vector<c10::Event> events{};
};

class AsyncSparseAllreduceCUDAWork : public AsyncSparseAllreduceWork {
 public:
  AsyncSparseAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncSparseAllreduceWork(context, inputs, tag, seq, timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to CPU tensors.
    // Note that both coalescing the sparse tensor and copying it to CPU
    // memory must be performed asynchronously, or we block the caller.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(
          inputs[i].coalesce().to(at::DeviceType::CPU, /*non_blocking=*/true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    auto output = allreduce(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(output, /*non_blocking=*/true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp{};
  std::vector<c10::Stream> streams{};
  std::vector<c10::Event> events{};
};

static c10::intrusive_ptr<ProcessGroupGloo::AsyncWork> makeAllreduceCUDAWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout) {
  auto layout = inputs[0].layout();

  if (layout == c10::kStrided) {
    if (context->getDevice()->hasGPUDirect()) {
      return c10::make_intrusive<AsyncAllreduceCUDADeviceWork>(
          std::move(context), inputs, reduceOp, tag, seq, timeout);
    } else {
      return c10::make_intrusive<AsyncAllreduceCUDAHostWork>(
          std::move(context), inputs, reduceOp, tag, seq, timeout);
    }
  } else if (layout == c10::kSparse) {
    return c10::make_intrusive<AsyncSparseAllreduceCUDAWork>(
        std::move(context), inputs, tag, seq, timeout);
  } else {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: unsupported layout");
  }
}

C10_REGISTER_TYPED_CREATOR(
    GlooAllreduceRegistry,
    at::kCUDA,
    makeAllreduceCUDAWork)
} // namespace c10d

#endif // USE_C10D_GLOO

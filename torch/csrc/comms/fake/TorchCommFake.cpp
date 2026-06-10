// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/TorchCommFactory.hpp>
#include <torch/csrc/comms/TorchWork.hpp>
#include <torch/csrc/comms/fake/TorchCommFake.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu

namespace torch::comms {

namespace {
class FakeTorchCommWindow : public TorchCommWindow {
 public:
  void tensor_register(const at::Tensor& tensor, bool owning = true) override {
    (void)tensor;
    (void)owning;
  }
  void tensor_deregister() override {}
  c10::intrusive_ptr<TorchWork> put(
      const at::Tensor& tensor,
      int dstRank,
      size_t targetOffsetNelems,
      bool asyncOp,
      const PutOptions& options) override {
    (void)tensor;
    (void)dstRank;
    (void)targetOffsetNelems;
    (void)asyncOp;
    (void)options;
    return c10::make_intrusive<TorchWorkCompleted>();
  }
  at::Tensor map_remote_tensor(int rank) override {
    (void)rank;
    return at::Tensor();
  }
  c10::intrusive_ptr<TorchWork> signal(
      int peerRank,
      bool asyncOp,
      const SignalOptions& options) override {
    (void)peerRank;
    (void)asyncOp;
    (void)options;
    return c10::make_intrusive<TorchWorkCompleted>();
  }
  c10::intrusive_ptr<TorchWork> wait_signal(
      int peerRank,
      bool asyncOp,
      const WaitSignalOptions& options) override {
    (void)peerRank;
    (void)asyncOp;
    (void)options;
    return c10::make_intrusive<TorchWorkCompleted>();
  }

  std::shared_ptr<TorchCommWindowAttr> get_attr(int peerRank) override {
    (void)peerRank;
    return nullptr;
  }

  std::shared_ptr<TorchCommWindow> clone() override {
    return std::make_shared<FakeTorchCommWindow>();
  }
};

class TorchWorkFailed : public TorchWork {
 public:
  TorchWorkFailed() {
    setStatus(WorkStatus::ERROR);
  }
  void wait() override {}
  void waitBlocking() override {}
};
} // namespace

TorchCommFake::TorchCommFake()
    : initialized_(false), device_(at::kCPU), rank_(0), size_(1) {}

void TorchCommFake::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  device_ = device;
  options_ = options;
  initialized_ = true;
  name_ = name;
}

void TorchCommFake::finalize() {
  initialized_ = false;
}

int TorchCommFake::getRank() const {
  return rank_;
}

int TorchCommFake::getSize() const {
  return size_;
}

std::string_view TorchCommFake::getCommName() const {
  return name_;
}

std::string_view TorchCommFake::getBackendName() const {
  return kBackendName;
}

c10::intrusive_ptr<TorchWork> TorchCommFake::send(
    const at::Tensor& /* tensor */,
    int /* dst */,
    bool /* async_op */,
    const SendOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::recv(
    at::Tensor& /* tensor */,
    int /* src */,
    bool /* async_op */,
    const RecvOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& /* ops */,
    bool /* async_op */,
    const BatchP2POptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::broadcast(
    at::Tensor& /* tensor */,
    int /* root */,
    bool /* async_op */,
    const BroadcastOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_reduce(
    at::Tensor& /* tensor */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const AllReduceOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::reduce(
    const at::Tensor& /* tensor */,
    int /* root */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_gather(
    const std::vector<at::Tensor>& /* tensor_list */,
    const at::Tensor& /* tensor */,
    bool /* async_op */,
    const AllGatherOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_gather_v(
    const std::vector<at::Tensor>& /* tensor_list */,
    const at::Tensor& /* tensor */,
    bool /* async_op */,
    const AllGatherOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_gather_single(
    at::Tensor& /* output */,
    const at::Tensor& /* input */,
    bool /* async_op */,
    const AllGatherSingleOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::reduce_scatter(
    at::Tensor& /* output */,
    const std::vector<at::Tensor>& /* input_list */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceScatterOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::reduce_scatter_v(
    at::Tensor& /* output */,
    const std::vector<at::Tensor>& /* input_list */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceScatterOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::reduce_scatter_single(
    at::Tensor& /* output */,
    const at::Tensor& /* input */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceScatterSingleOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_to_all_single(
    at::Tensor& /* output */,
    const at::Tensor& /* input */,
    bool /* async_op */,
    const AllToAllSingleOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_to_all_v_single(
    at::Tensor& /* output */,
    const at::Tensor& /* input */,
    const std::vector<uint64_t>& /* output_split_sizes */,
    const std::vector<uint64_t>& /* input_split_sizes */,
    bool /* async_op */,
    const AllToAllvSingleOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::all_to_all(
    const std::vector<at::Tensor>& /* output_tensor_list */,
    const std::vector<at::Tensor>& /* input_tensor_list */,
    bool /* async_op */,
    const AllToAllOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::barrier(
    bool /* async_op */,
    const BarrierOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::scatter(
    at::Tensor& /* output_tensor */,
    const std::vector<at::Tensor>& /* input_tensor_list */,
    int /* root */,
    bool /* async_op */,
    const ScatterOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

c10::intrusive_ptr<TorchWork> TorchCommFake::gather(
    const std::vector<at::Tensor>& /* output_tensor_list */,
    const at::Tensor& /* input_tensor */,
    int /* root */,
    bool /* async_op */,
    const GatherOptions& /* options */) {
  return c10::make_intrusive<TorchWorkCompleted>();
}

std::shared_ptr<TorchCommWindow> TorchCommFake::new_window(
    const std::optional<at::Tensor>& tensor) {
  auto win = std::make_shared<FakeTorchCommWindow>();
  if (tensor.has_value()) {
    win->tensor_register(tensor.value());
  }
  return win;
}

c10::intrusive_ptr<TorchWork> TorchCommFake::reconfigure(
    const ReconfigureOptions& /* opts */) {
  if (shouldFailReconfigure_) {
    initialized_ = false;
    return c10::make_intrusive<TorchWorkFailed>();
  }
  initialized_ = true;
  return c10::make_intrusive<TorchWorkCompleted>();
}

std::shared_ptr<TorchCommBackend> TorchCommFake::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  (void)ranks;
  (void)name;
  (void)options;
  return std::make_shared<TorchCommFake>();
}

const CommOptions& TorchCommFake::getOptions() const {
  return options_;
}

const at::Device& TorchCommFake::getDevice() const {
  return device_;
}

namespace {
class FakeRegistration {
 public:
  FakeRegistration() {
    TorchCommFactory::get().register_backend(
        "fake", []() { return std::make_shared<TorchCommFake>(); });
  }
};

static const FakeRegistration registration{};
} // namespace

} // namespace torch::comms

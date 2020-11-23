#include <c10d/ProcessGroupUCC.hpp>
#include <c10d/UCCSendRecv.hpp>
#include <utility>
#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#endif
#include <cstdio>

namespace c10d {

void ProcessGroupUCC::check_tensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error("ProcessGroupUCC takes 1 tensor");
  }
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error(
        "ProcessGroupUCC input tensor has to be contiguous");
  }
  if (tensors[0].is_sparse()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be dense");
  }
  // TODO: check cuda case
}

static torch_ucc_status_t compute_lengths_offsets(
    int group_size,
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    uint32_t* lengths,
    uint32_t* offsets) {
  bool equal_splits = false;
  size_t dim0_size = tensor.size(0);
  size_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  size_t split_size = 0;
  size_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }

  for (int i = 0; i < group_size; i++) {
    size_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    if ((length > INT_MAX) || (offset > INT_MAX)) {
      return TORCH_UCC_ERROR;
    }
    lengths[i] = length;
    offsets[i] = offset;
    offset += length;
  }

  return TORCH_UCC_OK;
}

ProcessGroupUCC::WorkUCX::~WorkUCX() {
  if (req != nullptr) {
    torch_ucx_request_free(req);
  }
}

bool ProcessGroupUCC::WorkUCX::isCompleted() {
  torch_ucx_status_t st;

  st = torch_ucx_req_test(comm, &req, 1, nullptr, 1, 1);
  return (st != TORCH_UCX_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCX::isSuccess() const {
  // TODO
  return true;
}

bool ProcessGroupUCC::WorkUCX::wait(
  std::chrono::milliseconds timeout /* unused */) {
  torch_ucx_req_test(comm, &req, 1, nullptr, -1, 1);
  return true;
}

ProcessGroupUCC::WorkColl::~WorkColl() {
  if (coll_req != nullptr) {
    if (coll_ops.coll_test(coll_req) != TORCH_UCC_OK) {
      fprintf(
          stderr,
          "ProcessGroupUCC: warn removing request before collective finish\n");
    }
    coll_ops.coll_finalize(coll_req);
  }

  if (scratch != nullptr) {
    delete[] scratch;
  }
}

bool ProcessGroupUCC::WorkColl::isCompleted() {
  torch_ucc_status_t st;

  if (!external_progress) {
    coll_ops.coll_progress(coll_req);
    st = coll_ops.coll_test(coll_req);
    if (st != TORCH_UCC_INPROGRESS) {
      work_list.erase(work_list_entry);
    }
  } else {
    st = coll_ops.coll_test(coll_req);
  }

  return (st != TORCH_UCC_INPROGRESS);
}

bool ProcessGroupUCC::WorkColl::isSuccess() const {
  // TODO
  return true;
}

bool ProcessGroupUCC::WorkColl::wait(
  std::chrono::milliseconds timeout /* unused */) {
  while (!isCompleted()) {
  };

  return true;
}

void ProcessGroupUCC::read_config() {
  char* env;

  config.enable_progress_thread = true;
  env = std::getenv("TORCH_UCC_THREAD_ENABLE");
  if (env) {
    config.enable_progress_thread = std::atoi(env);
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(std::move(store)), stop_progress_loop(false) {
  torch_ucx_status_t st;
  torch_ucc_status_t st_ucc;

  read_config();
  st = torch_ucx_comm_init(&ucx_comm, size, rank, store_);
  if (st != TORCH_UCX_OK) {
    throw std::runtime_error("ProcessGroupUCC init failed");
  }

  st_ucc = torch_ucc_coll_ops_init(&coll_ops);
  if (st_ucc != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC failed to init collops");
  }
  coll_comm = nullptr;

  if (config.enable_progress_thread) {
    progress_thread = std::thread(&ProcessGroupUCC::progress_loop, this);
  }
}

torch_ucc_coll_comm_t* ProcessGroupUCC::get_coll_comm() {
  if (coll_comm == nullptr) {
    torch_ucc_status_t st_ucc;

    st_ucc = coll_ops.coll_comm_init(ucx_comm, &coll_comm);
    if (st_ucc != TORCH_UCC_OK) {
      throw std::runtime_error(
          "ProcessGroupUCC failed to init collective comm");
    }
  }

  return coll_comm;
}

void ProcessGroupUCC::progress_loop() {
  std::unique_lock<std::mutex> lock(pg_mutex);
  torch_ucc_status_t st;

#ifdef USE_CUDA
  auto device = c10::Device(c10::DeviceType::CUDA, (c10::DeviceIndex)0);
  at::cuda::OptionalCUDAGuard guard(device);
  cudaSetDevice(0);
#endif

  while (!stop_progress_loop) {
    if (progress_list.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    auto work_coll = progress_list.front();
    progress_list.pop_front();
    lock.unlock();
    queue_consume_cv.notify_one();
#ifdef USE_CUDA
    if (work_coll->coll_req->dev_type == c10::DeviceType::CUDA) {
      guard.set_index(work_coll->coll_req->dev_index);
    }
#endif
    do {
      st = coll_ops.coll_progress(work_coll->coll_req);
    } while (
        (coll_ops.coll_test(work_coll->coll_req) == TORCH_UCC_INPROGRESS) &&
        (st == TORCH_UCC_OK));
    if (st != TORCH_UCC_OK) {
      fprintf(stderr, "ProcessGroupUCC: coll progress failed\n");
    }
    lock.lock();
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::enqueue_request(
    torch_ucc_coll_request_t* req,
    void* scratch) {
  std::unique_lock<std::mutex> lock(pg_mutex);

  auto iter = progress_list.emplace(
      progress_list.end(),
      c10::make_intrusive<ProcessGroupUCC::WorkColl>(coll_ops, progress_list));
  (*iter)->work_list_entry = iter;
  (*iter)->coll_req = req;
  (*iter)->external_progress = config.enable_progress_thread;
  (*iter)->scratch = (char*)scratch;
  auto workreq = (*iter);
  lock.unlock();
  queue_produce_cv.notify_one();
  return workreq;
}

ProcessGroupUCC::~ProcessGroupUCC() {
  if (config.enable_progress_thread) {
    std::unique_lock<std::mutex> lock(pg_mutex);
    queue_consume_cv.wait(lock, [&] { return progress_list.empty(); });
    stop_progress_loop = true;
    lock.unlock();
    queue_produce_cv.notify_all();
    progress_thread.join();
  }
  if (progress_list.size() != 0) {
    fprintf(stderr, "ProcessGroupUCC: warnning progress list is not empty\n");
  }
  coll_ops.coll_comm_close(coll_comm);
  torch_ucx_comm_close(ucx_comm, store_);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  ucc_comm = get_coll_comm();
  st = coll_ops.broadcast(ucc_comm, tensors, opts.rootRank, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: broadcast failed");
  }
  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  ucc_comm = get_coll_comm();
  st = coll_ops.allreduce(ucc_comm, tensors, opts, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: allreduce failed");
  }

  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& tensors /* unused */,
    const AllreduceCoalescedOptions& opts /* unused */) {
  throw std::runtime_error(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& tensors /* unused */,
    const ReduceOptions& opts /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts /* unused */) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  ucc_comm = get_coll_comm();
  st = coll_ops.allgather(ucc_comm, inputTensors, outputTensors[0], &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: allgather failed");
  }
  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(
    at::Tensor& outputBuffer /* unused */,
    at::Tensor& inputBuffer /* unused */,
    const AllgatherOptions& opts /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& opts /* unused */) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  ucc_comm = get_coll_comm();
  st = coll_ops.barrier(ucc_comm, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: barrier failed");
  }
  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors /* unused */,
    std::vector<at::Tensor>& inputTensors /* unused */,
    const GatherOptions& opts /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& outputTensors /* unused */,
    std::vector<std::vector<at::Tensor>>& inputTensors /* unused */,
    const ScatterOptions& opts /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& outputTensors /* unused */,
    std::vector<std::vector<at::Tensor>>& inputTensors /* unused */,
    const ReduceScatterOptions& opts /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts /* unused */) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;
  uint32_t* scratch;

  ucc_comm = get_coll_comm();
  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    scratch = nullptr;
    st = coll_ops.alltoall(ucc_comm, inputTensor, outputTensor, &coll_req);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoall_base failed");
    }
  } else {
    scratch = new uint32_t[4 * size_];
    uint32_t* send_lengths = scratch;
    uint32_t* recv_lengths = scratch + 1 * size_;
    uint32_t* send_offsets = scratch + 2 * size_;
    uint32_t* recv_offsets = scratch + 3 * size_;
    st = compute_lengths_offsets(
        size_, outputSplitSizes, outputTensor, recv_lengths, recv_offsets);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoallv failed");
    }
    st = compute_lengths_offsets(
        size_, inputSplitSizes, inputTensor, send_lengths, send_offsets);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoallv failed");
    }

    coll_ops.alltoallv(
        ucc_comm,
        inputTensor,
        send_lengths,
        send_offsets,
        outputTensor,
        recv_lengths,
        recv_offsets,
        &coll_req);
  }
  return enqueue_request(coll_req, scratch);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& outputTensors /* unused */,
    std::vector<at::Tensor>& inputTensors /* unused */,
    const AllToAllOptions& opts /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucx_status_t st;

  st = torch_ucx_send_nb(
      ucx_comm,
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      size,
      dstRank,
      tag,
      &req,
      TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to send msg");
  }

  return c10::make_intrusive<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucx_status_t st;

  st = torch_ucx_recv_nb(
      ucx_comm,
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      size,
      srcRank,
      tag,
      &req,
      TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to recv msg");
  }

  return c10::make_intrusive<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucx_status_t st;

  st = torch_ucx_recv_nb(
      ucx_comm,
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      size,
      TORCH_UCX_ANY_SOURCE,
      tag,
      &req,
      TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to recv msg");
  }

  return c10::make_intrusive<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

} // namespace c10d

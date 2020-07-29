#include <c10d/ProcessGroupUCX.hpp>

namespace c10d {

bool ProcessGroupUCX::WorkUCX::isCompleted()
{
  torch_ucx_status_t st;

  st = torch_ucx_req_test(comm, &req, 1, NULL, 1, 1);
  return (st != TORCH_UCX_INPROGRESS);
}

bool ProcessGroupUCX::WorkUCX::isSuccess() const
{
  if (req != NULL) {
    throw std::runtime_error(
      "Invalid call to isSuccess before work has completed");
  }
  return true;
}

bool ProcessGroupUCX::WorkUCX::wait(std::chrono::milliseconds)
{
  torch_ucx_req_test(comm, &req, 1, NULL, -1, 1);
  return true;
}

bool ProcessGroupUCX::WorkUCXColl::isCompleted()
{
  torch_ucx_status_t st;

  if (no_progress) {
    st = req->status;
  } else {
    st = torch_ucx_coll_test(req);
  }
  if (st != TORCH_UCX_INPROGRESS) {
    delete req;
  }

  return (st != TORCH_UCX_INPROGRESS);
}

bool ProcessGroupUCX::WorkUCXColl::isSuccess() const
{
  if (req != NULL) {
    throw std::runtime_error(
        "Invalid call to isSuccess before work has completed");
  }
  return true;
}

bool ProcessGroupUCX::WorkUCXColl::wait(std::chrono::milliseconds)
{
  bool completed;

  do {
    completed = isCompleted();
  } while (completed == false);

  return completed;
}

void ProcessGroupUCX::read_config()
{
    char *env;

    config.enable_progress_thread = true;
    env = std::getenv("TORCH_UCX_THREAD_ENABLE"); 
    if (env) {
        config.enable_progress_thread = std::atoi(env);
    }

}

ProcessGroupUCX::ProcessGroupUCX(const std::shared_ptr<Store>& store,
                                 int rank, int size,
                                 const std::chrono::milliseconds& opTimeout)
    : ProcessGroup(rank, size), store_(store), stop_progress_loop(false) {
  torch_ucx_status_t st;

  read_config();
  st = torch_ucx_comm_init(&ucx_comm, size, rank, store_);
  if (st != TORCH_UCX_OK) {
    throw std::runtime_error("ProcessGroupUCC init failed");
  }

  st = torch_ucx_coll_comm_init(ucx_comm, &ucx_coll_comm);
  if (st != TORCH_UCX_OK) {
    throw std::runtime_error("ProcessGroupUCC init failed");
  }

  if (config.enable_progress_thread) {
    progress_thread = std::thread(&ProcessGroupUCX::progress_loop, this);
  }
}

ProcessGroupUCX::~ProcessGroupUCX() {
  if (config.enable_progress_thread) {
    std::unique_lock<std::mutex> lock(pg_mutex);
    queue_consume_cv.wait(lock, [&] { return progress_queue.empty(); });
    stop_progress_loop = true;
    lock.unlock();
    queue_produce_cv.notify_all();
    progress_thread.join();
  }
  torch_ucx_coll_comm_close(ucx_coll_comm);
  torch_ucx_comm_close(ucx_comm, store_);
}

void ProcessGroupUCX::progress_loop()
{
    std::unique_lock<std::mutex> lock(pg_mutex);
    torch_ucx_coll_request_t     *req;
    torch_ucx_status_t           st;
 
    while(!stop_progress_loop) {
        if (progress_queue.empty()) {
            queue_produce_cv.wait(lock);
            continue;
        }
        req = progress_queue.front();
        progress_queue.pop_front();
        lock.unlock();
        queue_consume_cv.notify_one();
        do {
            st = torch_ucx_coll_test(req);
        } while(st == TORCH_UCX_INPROGRESS);
        lock.lock();
    }
}

void ProcessGroupUCX::enqueue_request(torch_ucx_coll_request_t* req)
{
    std::unique_lock<std::mutex> lock(pg_mutex);
    progress_queue.push_back(req);
    lock.unlock();
    queue_produce_cv.notify_one();
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::broadcast(
  std::vector<at::Tensor>& data,
  const BroadcastOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support broadcast");
  }

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support allreduce");
  }


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::allreduce_coalesced(
  std::vector<at::Tensor>& tensors,
  const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support allreduce_coalesced");
  }


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::reduce(
  std::vector<at::Tensor>& tensors,
  const ReduceOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support reduce");
  }


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::allgather(
  std::vector<std::vector<at::Tensor>>& outputTensors,
  std::vector<at::Tensor>& inputTensors,
  const AllgatherOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support allgather");
  }


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::allgather_base(
  at::Tensor& outputbuffer,
  at::Tensor& inputbuffer,
  const AllgatherOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support allgather_base");
  }

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::allgather_coalesced(
  std::vector<std::vector<at::Tensor>>& outputTensorLists,
  std::vector<at::Tensor>& inputTensors,
  const AllgatherOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support allgather_coalesced");
  }

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::gather(
  std::vector<std::vector<at::Tensor>>& outputTensors,
  std::vector<at::Tensor>& inputTensors,
  const GatherOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support gather");
  }


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::scatter(
  std::vector<at::Tensor>& outputTensors,
  std::vector<std::vector<at::Tensor>>& inputTensors,
  const ScatterOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support scatter");
  }

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::reduce_scatter(
  std::vector<at::Tensor>& outputTensors,
  std::vector<std::vector<at::Tensor>>& inputTensors,
  const ReduceScatterOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCX doesn't support reduce_scatter");
  }

int64_t computeLengthsAndOffsets(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    std::vector<int>* lengths,
    std::vector<int>* offsets) {
  int64_t group_size = lengths->size();
  bool equal_splits = false;
  int64_t dim0_size = tensor.size(0);
  int elem_size = tensor.element_size();
  int64_t row_size = elem_size * (dim0_size ? tensor.numel() / dim0_size : 1);
  int64_t split_size = 0;
  int64_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }
  for (int i = 0; i < group_size; i++) {
    int64_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    TORCH_INTERNAL_ASSERT(
        length <= std::numeric_limits<int>::max() &&
            offset <= std::numeric_limits<int>::max(),
        "Length or offset larger than INT_MAX not supported");
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    offset += length;
  }
  return offset;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::alltoall_base(
  at::Tensor& outputTensor,
  at::Tensor& inputTensor,
  std::vector<int64_t>& outputSplitSizes,
  std::vector<int64_t>& inputSplitSizes,
  const AllToAllOptions& opts) {

  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupUCX::alltoall_base: " + msg);
  };
  assertDense(invalidArgument, {outputTensor});
  assertDense(invalidArgument, {inputTensor});

  auto request = std::make_shared<ProcessGroupUCX::WorkUCXColl>();
  torch_ucx_coll_request_t *req = request->req;
  req->src_buf_mtype = inputTensor.is_cuda() ? TORCH_UCX_CUDA: TORCH_UCX_HOST;
  req->dst_buf_mtype = outputTensor.is_cuda() ? TORCH_UCX_CUDA: TORCH_UCX_HOST;
  req->src_buffer    = inputTensor.data_ptr();
  req->dst_buffer    = outputTensor.data_ptr();
  if ((outputSplitSizes.size() == 0) || (inputSplitSizes.size() == 0)) {
    req->len           = inputTensor.element_size() * inputTensor.numel() / size_;
    torch_ucx_alltoall_start(ucx_coll_comm, request->req);
  } else {
    req->send_lengths.resize(size_); req->send_offsets.resize(size_);
    req->recv_lengths.resize(size_); req->recv_offsets.resize(size_);
    computeLengthsAndOffsets(
        inputSplitSizes, inputTensor, &req->send_lengths, &req->send_offsets);
    computeLengthsAndOffsets(
        outputSplitSizes, outputTensor, &req->recv_lengths, &req->recv_offsets);
    torch_ucx_alltoallv_start(ucx_coll_comm, request->req);
  }

  if (config.enable_progress_thread) {
    fprintf(stderr, "using progress thread\n");
    enqueue_request(request->req);
    request->no_progress = true;
  }

  return request;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::alltoall(
  std::vector<at::Tensor>& outputTensors,
  std::vector<at::Tensor>& inputTensors,
  const AllToAllOptions& opts) {
  throw std::runtime_error(
    "ProcessGroupUCX doesn't support alltoall");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupUCX::send: " + msg);
  };
  assertSingleElement(invalidArgument, tensors);
  assertDense(invalidArgument, tensors);

  auto   &tensor = tensors[0];
  size_t size    = tensor.numel() * tensor.element_size();
  torch_ucx_request_t *req;
  torch_ucx_status_t  st;

  st = torch_ucx_send_nb(ucx_comm, tensor.data_ptr(), size, dstRank,
                         tag, &req, TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCX: failed to send msg");
  }

  return std::make_shared<ProcessGroupUCX::WorkUCX>(req, ucx_comm);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupUCX::send: " + msg);
  };
  assertSingleElement(invalidArgument, tensors);
  assertDense(invalidArgument, tensors);

  auto   &tensor = tensors[0];
  size_t size    = tensor.numel() * tensor.element_size();
  torch_ucx_request_t *req;
  torch_ucx_status_t  st;

  st = torch_ucx_recv_nb(ucx_comm, tensor.data_ptr(), size, srcRank,
                         tag, &req, TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCX: failed to recv msg");
  }

  return std::make_shared<ProcessGroupUCX::WorkUCX>(req, ucx_comm);
}


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::recvAnysource(
  std::vector<at::Tensor>& tensor,
  int tag) {
  throw std::runtime_error(
    "ProcessGroupUCX doesn't support recvAnysource");
}


std::shared_ptr<ProcessGroup::Work> ProcessGroupUCX::barrier(
  const BarrierOptions& opts) {
  throw std::runtime_error(
    "ProcessGroupUCX doesn't support barrier");
  }

}

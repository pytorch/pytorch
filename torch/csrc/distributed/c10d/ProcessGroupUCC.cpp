#ifdef USE_C10D_UCC

#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/UCCTracing.hpp>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace c10d {

namespace {
constexpr int64_t kBusyWaitMillis = 10;

const std::map<c10::DeviceType, ucc_memory_type_t> ucc_mtype_map = {
    {c10::kCPU, UCC_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCC_MEMORY_TYPE_CUDA},
};

ucc_memory_type_t to_ucc_memType(c10::DeviceType _c10_type) {
  if (ucc_mtype_map.find(_c10_type) != ucc_mtype_map.end())
    return ucc_mtype_map.at(_c10_type);
  else
    return UCC_MEMORY_TYPE_UNKNOWN;
}

const std::map<at::ScalarType, ucc_datatype_t> ucc_dtype_map = {
    {at::kByte, UCC_DT_UINT8},
    {at::kChar, UCC_DT_INT8},
    {at::kHalf, UCC_DT_FLOAT16},
    {at::kBFloat16, UCC_DT_BFLOAT16},
    {at::kDouble, UCC_DT_FLOAT64},
    {at::kFloat, UCC_DT_FLOAT32},
    {at::kInt, UCC_DT_INT32},
    {at::kLong, UCC_DT_INT64},
    {at::kBool, UCC_DT_UINT8},
};

ucc_datatype_t to_ucc_dType(at::Tensor _tensor) {
  if (_tensor.scalar_type() == at::kBool && _tensor.element_size() != 1) {
    TORCH_CHECK(
        false, "Size of Boolean type larger than 1 is not supported in UCC");
  }
  try {
    return ucc_dtype_map.at(_tensor.scalar_type());
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(false, "Not supported data type for UCC");
  }
}

const std::map<ReduceOp, ucc_reduction_op_t> ucc_op_map = {
    {ReduceOp::SUM, UCC_OP_SUM},
    {ReduceOp::PRODUCT, UCC_OP_PROD},
    {ReduceOp::MIN, UCC_OP_MIN},
    {ReduceOp::MAX, UCC_OP_MAX},
    {ReduceOp::BAND, UCC_OP_BAND},
    {ReduceOp::BOR, UCC_OP_BOR},
    {ReduceOp::BXOR, UCC_OP_BXOR},
    {ReduceOp::AVG, UCC_OP_AVG},
};

ucc_reduction_op_t to_ucc_reduceOp(
    const ReduceOp _op,
    const at::ScalarType _dt) {
  if (_dt == at::kBool) {
    if (_op == ReduceOp::SUM) {
      // bitwise or
      return UCC_OP_MAX;
    } else if (_op == ReduceOp::PRODUCT) {
      // bitwise and
      return UCC_OP_MIN;
    } else if (_op == ReduceOp::AVG) {
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with boolean inputs");
    }
  }

  try {
    return ucc_op_map.at(_op);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(false, "Not supported ReduceOp for UCC");
  }
}

struct torch_ucc_config_t {
  c10::once_flag flag;
  std::array<bool, 32> blocking_wait;
  bool enable_comms_logger;
  bool use_future;
  // Sharing UCC communicator among multiple PGs to save resource.
  bool shared_comm;
  // Using allgatherv to achieve allgather, without flattening the list of
  // (potentially non-contiguous) tensors.
  bool use_allgatherv;
  bool enable_health_check;
} torch_ucc_config;

std::unordered_map<std::string, std::string> torch_ucc_envs_map = {
    // TORCH_UCC_BLOCKING_WAIT allowed syntax:
    // - TORCH_UCC_BLOCKING_WAIT=none --> blocking wait completely disabled
    // - TORCH_UCC_BLOCKING_WAIT=all --> blocking wait completely enabled
    // - TORCH_UCC_BLOCKING_WAIT=allreduce,send,recv --> blocking wait enabled
    //                                                   on selected operations
    // Supported operations:
    // [allgather,allgather_base,allreduce,alltoall,broadcast,
    //  gather,reduce,reduce_scatter,scatter,send,recv]
    {"TORCH_UCC_BLOCKING_WAIT", "none"},

    {"TORCH_UCC_USE_FUTURE", "1"},
    {"TORCH_UCC_PROFILING_ENABLE", "0"},
    {"TORCH_UCC_SHARED_COMM", "1"},
    {"TORCH_UCC_USE_ALLGATHERV", "0"},
    {"TORCH_UCC_ENABLE_HEALTH_CHECK", "0"},
    {"TORCH_UCC_ENABLE_COMMS_LOGGER", "0"},
};

std::vector<OpType> parse_blocking_wait(std::string op_list_string) {
  const static std::unordered_map<std::string, OpType> str2op = {
      {"allgather", OpType::ALLGATHER},
      {"allgather_base", OpType::_ALLGATHER_BASE},
      {"allreduce", OpType::ALLREDUCE},
      {"alltoall_base", OpType::ALLTOALL_BASE},
      {"broadcast", OpType::BROADCAST},
      {"gather", OpType::GATHER},
      {"reduce", OpType::REDUCE},
      {"reduce_scatter", OpType::REDUCE_SCATTER},
      {"scatter", OpType::SCATTER},
      {"send", OpType::SEND},
      {"recv", OpType::RECV},
  };
  auto op_list = parse_list(op_list_string);
  if (op_list == std::vector<std::string>{"none"}) {
    return {};
  }
  std::vector<OpType> result;
  if (op_list == std::vector<std::string>{"all"}) {
    for (auto entry : str2op) {
      result.push_back(entry.second);
    }
  } else {
    for (auto op_string : op_list) {
      result.push_back(str2op.at(op_string));
    }
  }
  return result;
}

} // namespace

void read_config() {
  // default configuration
  torch_ucc_config.blocking_wait.fill(false);
  torch_ucc_config.use_future = true;
  torch_ucc_config.shared_comm = false;
  torch_ucc_config.use_allgatherv = false;
  torch_ucc_config.enable_health_check = false;
  torch_ucc_config.enable_comms_logger = false;

  // read all torch_ucc env. variables and update the map
  char* env;
  for (auto& torch_ucc_env : torch_ucc_envs_map) {
    env = std::getenv(torch_ucc_env.first.c_str());
    if (env) {
      torch_ucc_envs_map[torch_ucc_env.first] = std::string(env);
    }
  }

  auto blocking_wait_str = torch_ucc_envs_map.at("TORCH_UCC_BLOCKING_WAIT");
  for (auto op : parse_blocking_wait(blocking_wait_str)) {
    torch_ucc_config.blocking_wait[(std::uint8_t)op] = true;
  }
  // barrier is always blocking
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::BARRIER] = true;

  // barrier is always blocking
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::BARRIER] = true;

  torch_ucc_config.use_future =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_USE_FUTURE"));
  torch_ucc_config.shared_comm =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_SHARED_COMM"));
  torch_ucc_config.use_allgatherv =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_USE_ALLGATHERV"));
  torch_ucc_config.enable_health_check =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ENABLE_HEALTH_CHECK"));
  torch_ucc_config.enable_comms_logger =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ENABLE_COMMS_LOGGER"));
}

void check_device(c10::Device dev1, c10::Device dev2) {
  if (dev1.is_cuda() && dev2.is_cuda() && dev1 != dev2) {
    throw std::runtime_error("ProcessGroupUCC multidevice is not supported");
  }
}

void check_tensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "ProcessGroupUCC takes 1 tensor. Got " +
        std::to_string(tensors.size()) + ". ");
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

ProcessGroupUCC::WorkUCC::~WorkUCC() {
#ifdef USE_CUDA
  if (fence && ep) {
    std::lock_guard<std::mutex> lock(ep->event_pool_mutex);
    ep->event_pool.push(std::move(fence));
  }
#endif
}

void ProcessGroupUCC::WorkUCC::setException() {
  if (exception() || !entry_) {
    return;
  }
  exception_ = entry_->eptr_;
}

void ProcessGroupUCC::WorkUCC::setAndThrowException() {
  setException();
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

bool ProcessGroupUCC::WorkUCC::isCompleted() {
  if (!entry_) {
    return true;
  }
  setException();
  // status_ <= 0 to avoid listing all possible status codes.  The main thread
  // needs to be unblocked when UCC (in progress thread) returns success (== 0)
  // or any error code (< 0).
  return exception() || entry_->status_ <= 0;
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const {
  if (!entry_) {
    return true;
  }
  return !exception() && entry_->status_ == 0;
}

bool ProcessGroupUCC::WorkUCC::wait(std::chrono::milliseconds /* unused */) {
  if (torch_ucc_config.enable_comms_logger && logger_) {
    logger_->trace_generator->recordComms("wait", (uintptr_t)this, rank_);
  }
#ifdef USE_CUDA
  if (fence && !torch_ucc_config.blocking_wait[(int)opType_]) {
    // block user stream
    setAndThrowException();
    fence->block(at::cuda::getCurrentCUDAStream());
    return true;
  }
#endif
  // wait for complete.  For blocking case, the main thread will be blocked in
  // this loop until the progress thread changes the status of this request.
  // If timeout occurs, UCC will return UCC_ERR_TIMEOUT as the status.  The
  // main thread will throw out the exception then. There is no "abort"
  // function in UCC currently.
  while (!isCompleted())
    ;
  setAndThrowException();
  // manually call profiling end callbacks if they are set,
  // since progress thread does not own WorkUCC
  if (Work::recordFunctionEndCallback_) {
    Work::recordFunctionEndCallback_();
    Work::recordFunctionEndCallback_ = nullptr;
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupUCC::WorkUCC::getFuture() {
  return future_;
}

int ProcessGroupUCC::WorkUCC::sourceRank() const {
  if (opType_ != OpType::RECV && opType_ != OpType::RECVANYSOURCE) {
    // Throw an error
    return Work::sourceRank();
  }
  return sourceRank_;
}

std::vector<at::Tensor> ProcessGroupUCC::WorkUCC::result() {
  return *outputs_;
}

void ProcessGroupUCC::ProgressEntry::finalize(std::exception_ptr eptr) {
  ucc_status_t status = UCC_OK;

  if (request_ != nullptr) {
    status = request_->status;
    comm_->free_request(request_);
  }
  if (eptr) {
    eptr_ = eptr;
  } else {
    status_ = status;
  }
  if (future_) {
    if (eptr) {
      future_->setError(eptr);
    } else {
      future_->markCompleted(
          c10::IValue(data ? data->dst : std::vector<at::Tensor>()));
    }
  }
}

Comm::Comm(
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger_,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob_,
    c10::Device dev,
    bool is_health_check)
    : logger(logger_),
      oob(oob_),
      ucc_comm(oob, logger),
      finalize_phase(
          is_health_check ? TORCH_UCC_HEALTH_CHECK : TORCH_UCC_FINALIZE),
      cuda_device_index(TORCH_UCC_DEVICE_NOT_SET) {
  if (dev.is_cuda()) {
    cuda_device_index = dev.index();
  }
  stop_progress_loop = false;
  collective_inprogress = false;
  progress_thread = std::thread(&Comm::progress_loop, this);
#ifdef _GNU_SOURCE
  pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
#endif
}

Comm::~Comm() {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
}

std::shared_ptr<Comm> Comm::get_comm(
    uint32_t& id,
    c10::Device dev,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
    bool is_health_check) {
  static std::mutex m;
  static std::weak_ptr<Comm> comm;
  static uint32_t comm_id;

  std::lock_guard<std::mutex> lock(m);
  id = comm_id;

  std::string group_id = "group_id";
  if (is_health_check) {
    group_id = c10::str(dev.type()) + "/" + group_id;
  }

  std::vector<uint8_t> remote_comm_id;
  oob->store->deleteKey(group_id + std::to_string(0));
  if (oob->rank != 0) {
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&id),
        reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set(group_id + std::to_string(oob->rank), val);
  } else {
    for (int i = 1; i < oob->size; i++) {
      remote_comm_id = oob->store->get(group_id + std::to_string(i));
      oob->store->deleteKey(group_id + std::to_string(i));
      // Find the highest id.
      id = std::max(id, *(reinterpret_cast<uint32_t*>(remote_comm_id.data())));
    }
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&id),
        reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set(group_id + std::to_string(oob->rank), val);
  }
  remote_comm_id = oob->store->get(group_id + std::to_string(0));
  oob->comm_id = *(reinterpret_cast<uint32_t*>(remote_comm_id.data()));
  // Prepare comm_id (static variable) to the next id.
  comm_id = oob->comm_id + 1;

  if (torch_ucc_config.shared_comm) {
    std::shared_ptr<Comm> shared_comm = comm.lock();
    if (!shared_comm) {
      shared_comm = std::make_shared<Comm>(logger, oob, dev, is_health_check);
      comm = shared_comm;
    } else {
      if (dev.is_cuda() && !is_health_check) {
        if ((shared_comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
            (shared_comm->cuda_device_index != dev.index())) {
          TORCH_UCC_LOG_ERROR(
              is_health_check ? TORCH_UCC_HEALTH_CHECK : TORCH_UCC_INIT,
              "ucc communicator was initialized with different cuda device,"
              "multi device is not supported");
          throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
        }
        shared_comm->cuda_device_index = dev.index();
      }
    }
    return shared_comm;
  } else {
    return std::make_shared<Comm>(logger, oob, dev, is_health_check);
  }
}

void Comm::ucc_create_team(
    ucc_team_h& team,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
      UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = oob.get();
  team_params.oob.n_oob_eps = oob->size;
  team_params.oob.oob_ep = oob->rank;
  team_params.ep = oob->rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  TORCH_UCC_CHECK(
      ucc_team_create_post(&ucc_comm.context, 1, &team_params, &team),
      "failed to post team create");
  do {
    st = ucc_team_create_test(team);
    ucc_context_progress(ucc_comm.context);
  } while (st == UCC_INPROGRESS);
  TORCH_UCC_CHECK(st, "failed to create UCC team");
}

void Comm::ucc_destroy_team(ucc_team_h& team) {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });

  ucc_status_t status;
  while (UCC_INPROGRESS == (status = ucc_team_destroy(team))) {
    if (UCC_OK != status) {
      TORCH_UCC_LOG_ERROR(
          finalize_phase,
          c10::str("ucc team destroy error: ", ucc_status_string(status)));
      break;
    }
  }

  lock.unlock();
}

void Comm::enqueue_collective(
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
    ucc_coll_args_t& coll,
    ucc_team_h team) {
  ucc_coll_req_h request;
  TORCH_UCC_CHECK(
      ucc_collective_init(&coll, &request, team), "failed to init collective");
  TORCH_UCC_CHECK(ucc_collective_post(request), "failed to post collective");

  auto entry =
      std::make_shared<ProcessGroupUCC::ProgressEntry>(&ucc_comm, request);
  entry->data = std::move(data);
  entry->future_ = work->getFuture();
  work->entry_ = entry;
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(entry);
  lock.unlock();
  queue_produce_cv.notify_one();
}

#ifdef USE_CUDA
void Comm::enqueue_cuda_collective(
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
    ucc_coll_args_t& coll,
    ucc_team_h team,
    ucc_ee_h ee) {
  ucc_coll_req_h request;
  TORCH_UCC_CHECK(
      ucc_collective_init(&coll, &request, team),
      "failed to init cuda collective");
  ucc_ev_t comp_ev, *post_ev;
  comp_ev.ev_type = UCC_EVENT_COMPUTE_COMPLETE;
  comp_ev.ev_context = nullptr;
  comp_ev.ev_context_size = 0;
  comp_ev.req = request;
  TORCH_UCC_CHECK(
      ucc_collective_triggered_post(ee, &comp_ev),
      "failed to post triggered collective");
  ucc_status_t st = ucc_ee_get_event(ee, &post_ev);
  TORCH_CHECK(st == UCC_OK && post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST);
  ucc_ee_ack_event(ee, post_ev);
  auto entry =
      std::make_shared<ProcessGroupUCC::ProgressEntry>(&ucc_comm, request);
  entry->data = std::move(data);
  work->entry_ = entry;
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(entry);
  lock.unlock();
  queue_produce_cv.notify_one();
}
#endif

void Comm::progress_loop() {
  std::unique_lock<std::mutex> lock(mutex);
#ifdef USE_CUDA
  bool device_set = false;
#endif
  while (!stop_progress_loop) {
    if (progress_queue.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    collective_inprogress = true;
    auto work = progress_queue.front();
    progress_queue.pop_front();
    lock.unlock();
#ifdef USE_CUDA
    if ((!device_set) && (cuda_device_index != TORCH_UCC_DEVICE_NOT_SET)) {
      c10::cuda::set_device(cuda_device_index);
      device_set = true;
    }
#endif
    std::exception_ptr eptr;
    try {
      while (work->request_->status > 0) {
        ucc_comm.progress();
      }
      if (work->request_->status < 0) {
        eptr = std::make_exception_ptr(
            std::runtime_error(ucc_status_string(work->request_->status)));
        std::string err_log = c10::str(
            "Failed to progress communication", // TODO: report exact op type or
                                                // id?
            ucc_status_string(work->request_->status));
        TORCH_UCC_LOG_ERROR(TORCH_UCC_COLL_PROGRESS, err_log);
      }
    } catch (...) {
      eptr = std::current_exception();
    }
    work->finalize(eptr);
    work = nullptr;
    collective_inprogress = false;
    queue_consume_cv.notify_one();
    lock.lock();
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    std::chrono::duration<float> timeout)
    : ProcessGroup(rank, size), timeout_(timeout) {
  c10::call_once(torch_ucc_config.flag, read_config);
  oob = std::make_shared<torch_ucc_oob_coll_info_t>();
  oob->rank = rank;
  oob->size = size;
  oob->store = store;
  comm = nullptr;
  cuda_ee = nullptr;
  static uint32_t id = 0;
  uint32_t pg_id = id++;

  logger = c10::make_intrusive<ProcessGroupUCCLogger>(
      c10::str("[Rank ", rank_, "]", "[ProcessGroupUCC-", pg_id, "]"),
      TORCH_UCC_INIT);
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_INIT,
      c10::str(
          "Created ProcessGroupUCC with ",
          size,
          " ranks, with timeout ",
          timeout_.count(),
          " secs"));
  std::string envs = "";
  for (auto& torch_ucc_env : torch_ucc_envs_map) {
    envs += ("\n\t" + torch_ucc_env.first + "=" + torch_ucc_env.second);
  }
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_INIT,
      c10::str(
          "Successfully read and set ProcessGroupUCC env. variables as followings",
          envs));

  if (torch_ucc_config.enable_health_check) {
    // Perform health check by initializing dummy communicators and destroying
    // them. This will help indicate any UCC/UCX-related issues prior to the
    // first collective. Run it in a separate thread and wait on CV to handle
    // timeouts so that if there are hangs, the main thread can still run
    // correctly.
    runHealthCheck();
  }
  if (torch_ucc_config.enable_comms_logger) {
    logger->initCommsTracer();
  }
}

ProcessGroupUCC::~ProcessGroupUCC() {
  if (torch_ucc_config.enable_comms_logger) {
    logger->flushComms(this->getRank(), this->getSize());
  }
  if (comm) {
    logger->setPhase(TORCH_UCC_FINALIZE);
    comm->ucc_destroy_team(team);
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_FINALIZE, "Successfully destroyed UCC library");
    try {
      if (cuda_ee) {
        ucc_ee_destroy(cuda_ee);
      }
    } catch (std::exception& ex) {
      TORCH_UCC_LOG_INFO(
          TORCH_UCC_FINALIZE,
          c10::str(
              "(~ProcessGroupUCC) Caught error in Store Operation .. ",
              "[",
              ex.what(),
              "]"));
    }
    comm = nullptr;
  }
}

#ifdef USE_CUDA
// Return CUDA device with ordinal given by input rank.
c10::Device getCUDADeviceForRank(int rank) {
  TORCH_CHECK(rank >= 0, "Invalid rank ", rank);
  auto numGPUs = at::cuda::getNumGPUs();
  auto deviceIdx = static_cast<c10::DeviceIndex>(rank % numGPUs);
  return c10::Device(c10::DeviceType::CUDA, deviceIdx);
}
#endif

void ProcessGroupUCC::runHealthCheck() {
  // Run health check in a separate thread and wait on CV to handle timeouts.
  // This design allows us to handle hangs.

  // When size_ is 1, there is no need to do any communication at all.
  if (size_ == 1)
    return;

  struct HealthCheckData {
    std::mutex healthCheckMutex;
    std::condition_variable healthCheckCv;
    bool uccHealthCheckSuccess = false;
    std::exception_ptr healthCheckException;
  } healthCheckData;

  auto t = std::thread([&healthCheckData, this]() {
    std::list<c10::Device> devices{c10::kCPU};
#ifdef USE_CUDA
    c10::cuda::OptionalCUDAGuard gpuGuard;
    if (at::cuda::is_available()) {
      devices.emplace_front(getCUDADeviceForRank(rank_));
    }
#endif
    for (auto device : devices) {
      bool is_last_device = (device == devices.back());
      try {
        auto oob = std::make_shared<torch_ucc_oob_coll_info_t>();
        oob->rank = this->oob->rank;
        oob->size = this->oob->size;
        oob->store = this->oob->store;
        ucc_team_h team = nullptr;
        uint32_t comm_id;
#ifdef USE_CUDA
        if (device.is_cuda()) {
          gpuGuard.set_index(device.index());
        }
#endif
        auto comm = Comm::get_comm(comm_id, device, oob, logger, true);
        comm->ucc_create_team(team, oob);
        comm->ucc_destroy_team(team);
        TORCH_UCC_LOG_INFO(
            TORCH_UCC_HEALTH_CHECK,
            c10::str(
                "UCC library health check succeed for device ",
                c10::DeviceTypeName(device.type())));
        // Mark ucc health check as complete.
        if (is_last_device) {
          std::lock_guard<std::mutex> lk(healthCheckData.healthCheckMutex);
          healthCheckData.uccHealthCheckSuccess = true;
        }

        comm = nullptr;
        oob = nullptr;
        // Notify main thread the health check is complete.
        if (is_last_device) {
          healthCheckData.healthCheckCv.notify_one();
        }
      } catch (const std::exception& e) {
        // Populate exception ptr.
        healthCheckData.healthCheckException = std::current_exception();
        // Unblock waiting main thread which will report exception.
        healthCheckData.healthCheckCv.notify_one();
      } // Unknown exceptions will just cause the program to terminate.
    }
  });
  // We don't need to join the thread, just need to verify health check via the
  // CV. Hence we detach the thread here.
  t.detach(); // NOLINT
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_HEALTH_CHECK,
      c10::str(
          "will wait up to ",
          timeout_.count(),
          " msec for UCC health check to complete."));
  std::unique_lock<std::mutex> lock(healthCheckData.healthCheckMutex);
  healthCheckData.healthCheckCv.wait_for(lock, timeout_, [&healthCheckData]() {
    return healthCheckData.uccHealthCheckSuccess;
  });

  if (healthCheckData.healthCheckException) {
    std::rethrow_exception(healthCheckData.healthCheckException);
  }
  // If there is no exception, the likely culprit is a timeout/hang
  TORCH_CHECK(
      healthCheckData.uccHealthCheckSuccess,
      "ProcessGroupUCC: Health check failure: Failed to initialize UCC on rank ",
      rank_);
}

void ProcessGroupUCC::set_timeout(ucc_coll_args_t& args) {
  args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
  args.flags |= UCC_COLL_ARGS_FLAG_TIMEOUT;
  args.timeout = timeout_.count();
}

#ifdef USE_CUDA
std::unique_ptr<at::cuda::CUDAEvent> ProcessGroupUCC::getPooledEvent() {
  std::unique_ptr<at::cuda::CUDAEvent> ev;
  std::lock_guard<std::mutex> lock(ep.event_pool_mutex);
  if (ep.event_pool.empty()) {
    ev = std::make_unique<at::cuda::CUDAEvent>();
  } else {
    ev = std::move(ep.event_pool.front());
    ep.event_pool.pop();
  }
  return ev;
}
#endif

template <typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupUCC::collective_post(
    OpType opType,
    PreProcess preproc,
    PostProcess postproc,
    ucc_coll_args_t& coll,
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::Device dev,
    std::vector<at::Tensor>& inputTensors,
    std::vector<at::Tensor>& outputTensors,
    const char* prof_title) {
  seq_++;
  set_timeout(coll);
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
      opType, seq_, prof_title, inputTensors, logger);

  if (opType == OpType::RECV) {
    work->sourceRank_ = coll.root;
  }

  RECORD_COMMS_TRACE(
      logger->trace_generator,
      work,
      opType,
      this->getRank(),
      this->getSize(),
      inputTensors,
      outputTensors);

  // Store references to outputs to be used by result
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputTensors);
  switch (dev.type()) {
    case c10::DeviceType::CPU: {
      if (torch_ucc_config.use_future) {
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
      }
      comm->enqueue_collective(std::move(data), work, coll, team);
      return work;
    }
#ifdef USE_CUDA
    case c10::DeviceType::CUDA: {
      auto cuda_ev = getPooledEvent();
      cuda_ev->record(at::cuda::getCurrentCUDAStream(dev.index()));
      cuda_ev->block(*stream);
      at::cuda::CUDAStreamGuard guard(*stream);
      preproc();
      comm->enqueue_cuda_collective(std::move(data), work, coll, team, cuda_ee);
      postproc();
      cuda_ev->record(*stream);
      work->fence = std::move(cuda_ev);
      work->ep = &ep;
      if (torch_ucc_config.use_future) {
        c10::cuda::CUDAMultiStreamGuard streamGuard(*stream);
        std::vector<c10::Device> devList{dev};
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()), devList);
        // Add a callback that runs profiling end callbacks
        if (work->recordFunctionEndCallback_) {
          work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          });
        }

        work->future_->markCompleted(c10::IValue(outputTensors));
      }
      return work;
    }
#endif // #ifdef USE_CUDA
    default: {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, c10::str("unsupported device type ", dev.str()));
      throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
    }
  }
}

c10::intrusive_ptr<Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  auto& tensor = inputTensors[0];
  check_device(tensor.device(), outputTensors[0][0].device());
  initComm(tensor.device());

  if (tensor.device().is_cpu() || torch_ucc_config.use_allgatherv) {
    AllgathervWorkData* data = new AllgathervWorkData(size_);
    for (int i = 0; i < size_; i++) {
      data->recv_lengths[i] = tensor.element_size() * tensor.numel();
      data->recv_offsets[i] = (uint64_t)outputTensors[0][i].data_ptr();
    }
    ucc_coll_args_t coll;
    coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll.flags =
        UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    coll.coll_type = UCC_COLL_TYPE_ALLGATHERV;
    coll.src.info.buffer = tensor.data_ptr();
    coll.src.info.count = tensor.element_size() * tensor.numel();
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
    coll.dst.info_v.buffer = nullptr;
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = UCC_DT_UINT8;
    coll.dst.info_v.mem_type =
        to_ucc_memType(outputTensors[0][0].device().type());
    SAVE_TENSORS(inputTensors, data->src);
    SAVE_TENSORS(outputTensors[0], data->dst);

    return collective_post(
        OpType::ALLGATHER,
        []() {},
        []() {},
        coll,
        std::unique_ptr<WorkData>(data),
        tensor.device(),
        inputTensors,
        outputTensors[0],
        "ucc:all_gather");
  } else {
    WorkData* data = new WorkData();
    std::vector<at::Tensor> flat_output(outputTensors.size());
    for (size_t i = 0; i < outputTensors.size(); i++) {
      TORCH_CHECK(
          outputTensors[i].size() == outputTensors.size() * size_,
          "Tensor output list is not valid for the number of participants");
      flat_output[i] = c10d::newLikeFlat(outputTensors, i);
    }
    SAVE_TENSORS(flat_output, data->flat);
    ucc_coll_args_t coll;
    coll.mask = 0;
    coll.flags = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLGATHER;
    coll.src.info.buffer = tensor.data_ptr();
    coll.src.info.count = tensor.numel();
    coll.src.info.datatype = to_ucc_dType(tensor);
    coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
    coll.dst.info.buffer = flat_output[0].data_ptr();
    coll.dst.info.count = flat_output[0].numel();
    coll.dst.info.datatype = to_ucc_dType(flat_output[0]);
    coll.dst.info.mem_type =
        to_ucc_memType(outputTensors[0][0].device().type());

    auto copy_from_flat = [&] {
      bool asyncCopy = false;
#ifdef USE_CUDA
      bool isCuda = outputTensors[0][0].device().is_cuda();
      ;
#endif
      for (size_t i = 0; i < outputTensors.size(); i++) {
        auto inumel = inputTensors[i].numel();
        for (size_t j = 0; j < outputTensors[i].size(); j++) {
          TORCH_CHECK(
              (outputTensors[i][j].numel() == inumel),
              "Tensor operand counts must be same");
#ifdef USE_CUDA
          if (isCuda) {
            c10::cuda::CUDACachingAllocator::recordStream(
                outputTensors[i][j].storage().data_ptr(), (*stream));
            asyncCopy = true;
          }
#endif
          outputTensors[i][j].copy_(flat_output[i][j], asyncCopy);
        }
      }
    };
    return collective_post(
        OpType::ALLGATHER,
        []() {},
        copy_from_flat,
        coll,
        std::unique_ptr<WorkData>(data),
        tensor.device(),
        inputTensors,
        outputTensors[0],
        "ucc:all_gather");
  }
}

c10::intrusive_ptr<Work> ProcessGroupUCC::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& opts) {
  check_tensor({outputTensor});
  check_tensor({inputTensor});
  initComm(outputTensor.device());

  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_ALLGATHER;
  coll.src.info.buffer = inputTensor.data_ptr();
  coll.src.info.count = inputTensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(inputTensor.scalar_type());
  coll.src.info.mem_type = to_ucc_memType(inputTensor.device().type());
  coll.dst.info.buffer = outputTensor.data_ptr();
  coll.dst.info.count = outputTensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(outputTensor.scalar_type());
  coll.dst.info.mem_type = to_ucc_memType(outputTensor.device().type());

  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::_ALLGATHER_BASE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      outputTensor.device(),
      inputTensors,
      outputTensors,
      "ucc:allgather_base");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_ALLREDUCE;
  coll.op = to_ucc_reduceOp(opts.reduceOp, tensor.scalar_type());
  coll.src.info.buffer = nullptr;
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = tensor.numel();
  coll.dst.info.datatype = to_ucc_dType(tensor);
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
  SAVE_TENSORS(tensors, data->dst);
  return collective_post(
      OpType::ALLREDUCE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      tensors,
      "ucc:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
  }

  initComm(device);
  ucc_coll_args_t coll;
  AlltoallWorkData* data;
  data = new AlltoallWorkData(size_);

  /* to avoid flatten the tensors, we use alltoallv to achieve Alltoall as
     follow.
      1. store addresses of each tensor directly in displacements, keep buffer
     to nullptr, i.e., 0
      2. convert datatype to UINT8, which is always 1 bytes, to avoid wrong size
     calculation in UCC layer
      3. post Alltoallv
  */
  for (const auto i : c10::irange(size_)) {
    data->send_lengths[i] =
        (uint64_t)(inputTensors[i].element_size() * inputTensors[i].numel());
    data->send_offsets[i] = (uint64_t)inputTensors[i].data_ptr();
    data->recv_lengths[i] =
        (uint64_t)(outputTensors[i].element_size() * outputTensors[i].numel());
    data->recv_offsets[i] = (uint64_t)outputTensors[i].data_ptr();
  }

  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
  coll.src.info_v.buffer = 0;
  coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
  coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
  coll.src.info_v.datatype = UCC_DT_UINT8;
  coll.src.info_v.mem_type = to_ucc_memType(inputTensors[0].device().type());
  coll.dst.info_v.buffer = 0;
  coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
  coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
  coll.dst.info_v.datatype = UCC_DT_UINT8;
  coll.dst.info_v.mem_type = to_ucc_memType(outputTensors[0].device().type());

  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::ALLTOALL,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      device,
      inputTensors,
      outputTensors,
      "ucc:alltoall");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_device(inputTensor.device(), outputTensor.device());
  initComm(inputTensor.device());
  ucc_coll_args_t coll;
  AlltoallWorkData* data;

  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    data = new AlltoallWorkData(0);
    TORCH_CHECK(
        (outputTensor.size(0) % size_ == 0) &&
            (inputTensor.size(0) % size_ == 0),
        "Tensor's dim 0 does not divide equally across group size");
    coll.mask = 0;
    coll.flags = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALL;
    coll.src.info.buffer = inputTensor.data_ptr();
    coll.src.info.count = inputTensor.element_size() * inputTensor.numel();
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = to_ucc_memType(inputTensor.device().type());
    coll.dst.info.buffer = outputTensor.data_ptr();
    coll.dst.info.count = outputTensor.element_size() * outputTensor.numel();
    coll.dst.info.datatype = UCC_DT_UINT8;
    coll.dst.info.mem_type = to_ucc_memType(outputTensor.device().type());
    coll.flags = 0;
  } else {
    data = new AlltoallWorkData(size_);
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    computeLengthsAndOffsets(
        outputSplitSizes,
        outputTensor,
        &data->recv_lengths,
        &data->recv_offsets);
    computeLengthsAndOffsets(
        inputSplitSizes, inputTensor, &data->send_lengths, &data->send_offsets);
    coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
    coll.src.info_v.buffer = inputTensor.data_ptr();
    coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
    coll.src.info_v.datatype = to_ucc_dType(inputTensor);
    coll.src.info_v.mem_type = to_ucc_memType(inputTensor.device().type());
    coll.dst.info_v.buffer = outputTensor.data_ptr();
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = to_ucc_dType(outputTensor);
    coll.dst.info_v.mem_type = to_ucc_memType(outputTensor.device().type());
    coll.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
        UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER | UCC_COLL_ARGS_FLAG_COUNT_64BIT |
        UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

    if (torch_ucc_config.enable_comms_logger) {
      logger->trace_generator->recordOptionalInfo(
          outputSplitSizes, inputSplitSizes);
    }
  }
  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::ALLTOALL_BASE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      inputTensor.device(),
      inputTensors,
      outputTensors,
      "ucc:alltoall");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::barrier(const BarrierOptions& opts) {
  c10::Device device = c10::Device(c10::DeviceType::CPU);
#ifdef USE_CUDA
  auto numGPUs = c10::cuda::device_count();
  if (!opts.device_ids.empty()) {
    device = c10::Device(c10::DeviceType::CUDA, opts.device_ids.front());
  } else if (comm && comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) {
    device = c10::Device(c10::DeviceType::CUDA, comm->cuda_device_index);
  } else if (numGPUs > 0) {
    int8_t deviceIdx = static_cast<int8_t>(c10::cuda::current_device());
    // if current device is 0, likely the device is not set, use the best guess
    if (0 == (int)deviceIdx) {
      deviceIdx = static_cast<int8_t>(this->getRank() % numGPUs);
    }
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_COLL_POST,
        c10::str(
            "post barrier before specifying any GPU while there are ",
            numGPUs,
            " GPUs available. ",
            "Not clear if GPU barrier is required, using GPU ",
            (int)deviceIdx,
            " to perform barrier. ",
            "Specify device_ids option in barrier() to force ",
            "use of a particular device"));
    device = c10::Device(c10::DeviceType::CUDA, deviceIdx);
  }
#endif
  initComm(device);

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BARRIER;
  auto dummy_tensor = std::vector<at::Tensor>();
  return collective_post(
      OpType::BARRIER,
      []() {},
      []() {},
      coll,
      nullptr,
      device,
      dummy_tensor,
      dummy_tensor,
      "ucc:barrier");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.root = opts.rootRank;
  SAVE_TENSORS(tensors, data->dst);

  if (torch_ucc_config.enable_comms_logger) {
    logger->trace_generator->recordOptionalInfo(opts.rootRank);
  }

  return collective_post(
      OpType::BROADCAST,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      tensors,
      "ucc:broadcast");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  std::vector<at::Tensor> outputs;
  auto& input = inputTensors[0];
  initComm(input.device());

  AllgathervWorkData* data = new AllgathervWorkData(size_);
  ucc_coll_args_t coll;
  coll.root = opts.rootRank;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_GATHERV;

  /* for non-root ranks, only src is valid */
  coll.src.info.buffer = input.data_ptr();
  coll.src.info.count = (uint64_t)(input.element_size() * input.numel());
  coll.src.info.datatype = UCC_DT_UINT8;
  coll.src.info.mem_type = to_ucc_memType(input.device().type());

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "gather requires a single-element output list containing a list with ",
              getSize(),
              " tensors."));
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "Incorrect output list size ",
              outputTensors[0].size(),
              ". Output list size should be ",
              getSize(),
              ", same as size of the process group."));
    }
    outputs = outputTensors[0];

    for (int i = 0; i < size_; i++) {
      data->recv_lengths[i] =
          (uint64_t)(outputs[i].element_size() * outputs[i].numel());
      data->recv_offsets[i] = (uint64_t)outputs[i].data_ptr();
    }
    /* use gatherv and store non-contiguous addresses in displacements to avoid
     * flatten outputTensors */
    coll.dst.info_v.buffer = nullptr;
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = UCC_DT_UINT8;
    coll.dst.info_v.mem_type = to_ucc_memType(outputs[0].device().type());

    SAVE_TENSORS(outputs, data->dst);
  } else {
    // for non-root ranks, outputTensors should be an empty list
    if (outputTensors.size() != 0) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, "requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list to be used by future mark
    outputs.emplace_back();
  }

  SAVE_TENSORS(inputTensors, data->src);

  return collective_post(
      OpType::GATHER,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      input.device(),
      inputTensors,
      outputs,
      "ucc:gather");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_REDUCE;
  coll.op = ucc_op_map.at(opts.reduceOp);
  coll.root = opts.rootRank;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = tensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
  SAVE_TENSORS(tensors, data->dst);
  return collective_post(
      OpType::REDUCE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      tensors,
      "ucc:reduce");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(
      (outputTensors.size() == inputTensors.size()),
      "Tensor input/output list for reduce_scatter must have same size");
  check_tensor(outputTensors);
  check_device(inputTensors[0][0].device(), outputTensors[0].device());
  initComm(inputTensors[0][0].device());
  auto data = std::make_unique<WorkData>();
  std::vector<at::Tensor> flat_input(inputTensors.size());
  for (size_t i = 0; i < inputTensors.size(); i++) {
    TORCH_CHECK(
        inputTensors[i].size() == inputTensors.size() * size_,
        "Tensor input list is not valid for the number of participants");
    flat_input[i] = c10d::newLikeFlat(inputTensors, i);
  }
  SAVE_TENSORS(flat_input, data->flat);
  check_tensor(flat_input);
  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;
  coll.op = to_ucc_reduceOp(opts.reduceOp, flat_input[0].scalar_type());

  coll.src.info.buffer = flat_input[0].data_ptr();
  coll.src.info.count = flat_input[0].numel();
  coll.src.info.datatype = to_ucc_dType(flat_input[0]);
  coll.src.info.mem_type = to_ucc_memType(flat_input[0].device().type());
  coll.dst.info.buffer = outputTensors[0].data_ptr();
  coll.dst.info.count = outputTensors[0].numel();
  coll.dst.info.datatype = to_ucc_dType(outputTensors[0]);
  coll.dst.info.mem_type = to_ucc_memType(outputTensors[0].device().type());

  SAVE_TENSORS(inputTensors[0], data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  auto copy_to_flat = [&] {
    bool asyncCopy = false;
    auto isize = inputTensors.size();
#ifdef USE_CUDA
    bool isCuda = inputTensors[0][0].device().is_cuda();
#endif
    for (size_t i = 0; i < isize; i++) {
      auto onumel = outputTensors[i].numel();
      for (size_t j = 0; j < inputTensors[i].size(); j++) {
        TORCH_CHECK(
            (inputTensors[i][j].numel() == onumel),
            "Tensor operand counts must be same");
#ifdef USE_CUDA
        if (isCuda) {
          c10::cuda::CUDACachingAllocator::recordStream(
              inputTensors[i][j].storage().data_ptr(), (*stream));
          asyncCopy = true;
        }
#endif
        flat_input[i][j].copy_(inputTensors[i][j], asyncCopy);
      }
    }
  };

  return collective_post(
      OpType::REDUCE_SCATTER,
      copy_to_flat,
      []() {},
      coll,
      std::move(data),
      inputTensors[0][0].device(),
      inputTensors[0],
      outputTensors,
      "ucc:reduce_scatter");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  auto& tensor = outputTensors[0];
  initComm(tensor.device());

  ScattervWorkData* data = new ScattervWorkData(size_);
  ucc_coll_args_t coll;
  coll.root = opts.rootRank;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_SCATTERV;

  if (getRank() == opts.rootRank) {
    /* src is only valid at non-root rank */
    if (inputTensors.size() != 1) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "gather requires a single-element output list containing a list with ",
              getSize(),
              " tensors."));
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "Incorrect output list size ",
              inputTensors[0].size(),
              ". Output list size should be ",
              getSize(),
              ", same as size of the process group."));
    }

    for (int i = 0; i < size_; i++) {
      data->send_lengths[i] = (uint64_t)tensor.element_size() * tensor.numel();
      data->send_offsets[i] = (uint64_t)inputTensors[0][i].data_ptr();
    }
    /* use scatter and store non-contiguous addresses in displacements to avoid
     * flatten inputTensors */
    coll.src.info_v.buffer = nullptr;
    coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
    coll.src.info_v.datatype = UCC_DT_UINT8;
    coll.src.info_v.mem_type =
        to_ucc_memType(inputTensors[0][0].device().type());

    SAVE_TENSORS(inputTensors[0], data->src);
  } else {
    // for non-root ranks, inputTensors should be an empty list
    if (inputTensors.size() != 0) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, "requires empty output on non-root");
    }
  }

  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = (uint64_t)tensor.element_size() * tensor.numel();
  coll.dst.info.datatype = UCC_DT_UINT8;
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::SCATTER,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      inputTensors[0],
      outputTensors,
      "ucc:scatter");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  WorkData* data = new WorkData();
  ucc_coll_args_t coll;
  coll.tag = tag;
  coll.mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.root = getRank();

  coll.active_set.size = 2;
  coll.active_set.start = getRank();
  coll.active_set.stride = dstRank - getRank();
  SAVE_TENSORS(tensors, data->dst);

  return collective_post(
      OpType::SEND,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      tensors,
      "ucc:send");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  WorkData* data = new WorkData();
  ucc_coll_args_t coll;
  coll.tag = tag;
  coll.mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.root = srcRank;

  coll.active_set.size = 2;
  coll.active_set.start = srcRank;
  coll.active_set.stride = getRank() - srcRank;
  SAVE_TENSORS(tensors, data->dst);

  return collective_post(
      OpType::RECV,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      tensors,
      "ucc:recv");
}

void ProcessGroupUCC::setSequenceNumberForGroup() {}

uint64_t ProcessGroupUCC::getSequenceNumberForGroup() {
  return seq_;
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size, timeout);
}

void ProcessGroupUCC::initComm(c10::Device dev) {
  if (!comm) {
#ifdef USE_CUDA
    if (dev.is_cuda()) {
      c10::cuda::set_device(dev.index());
    }
#endif
    comm = Comm::get_comm(comm_id, dev, oob, logger);
    TORCH_UCC_LOG_INFO(TORCH_UCC_INIT, "Successfully initialized UCX library");
    comm->ucc_create_team(team, oob);
    TORCH_UCC_LOG_INFO(TORCH_UCC_INIT, "Successfully initialized UCC library");
    logger->setPhase(TORCH_UCC_READY);
  } else {
    if (dev.is_cuda()) {
      if ((comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
          (comm->cuda_device_index != dev.index())) {
        TORCH_UCC_LOG_ERROR(
            TORCH_UCC_INIT,
            "ucc communicator was initialized with different cuda device,"
            "multi device is not supported");
        throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
      }
      comm->cuda_device_index = dev.index();
    }
  }
#ifdef USE_CUDA
  // Create UCC execution engine.
  if (!cuda_ee && dev.is_cuda()) {
    stream = std::make_unique<at::cuda::CUDAStream>(
        at::cuda::getStreamFromPool(true, dev.index()));
    ucc_ee_params_t params;
    params.ee_type = UCC_EE_CUDA_STREAM;
    params.ee_context = (void*)stream->stream();
    params.ee_context_size = sizeof(cudaStream_t);
    TORCH_UCC_CHECK(
        ucc_ee_create(team, &params, &cuda_ee),
        "failed to create UCC execution engine");
  }
#endif
}

} // namespace c10d

#endif // USE_C10D_UCC

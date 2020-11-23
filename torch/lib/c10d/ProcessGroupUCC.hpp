#pragma once

#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <c10d/UCCOps.hpp>
#include <c10d/UCCSendRecv.hpp>

namespace c10d {

class ProcessGroupUCC : public ProcessGroup {
 public:
  class WorkUCX : public ProcessGroup::Work {
   public:
    WorkUCX(torch_ucx_request_t* request, torch_ucx_comm_t* ucx_comm)
        : req(request), comm(ucx_comm) {}
    virtual ~WorkUCX() override;
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

   protected:
    torch_ucx_request_t* req;
    torch_ucx_comm_t* comm;
    friend class ProcessGroupUCC;
  };

  class WorkColl : public ProcessGroup::Work {
   public:
    WorkColl(
        torch_ucc_coll_ops_t ops,
        std::list<c10::intrusive_ptr<WorkColl>>& list)
        : coll_ops(ops),
          work_list(list),
          external_progress(false),
          scratch(nullptr) {}

    virtual ~WorkColl() override;
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

   protected:
    torch_ucc_coll_ops_t coll_ops;
    std::list<c10::intrusive_ptr<WorkColl>>& work_list;
    std::list<c10::intrusive_ptr<WorkColl>>::iterator work_list_entry;
    bool external_progress;
    char* scratch;
    std::vector<at::Tensor> src;
    std::vector<at::Tensor> dst;
    bool no_progress{};
    torch_ucc_coll_request_t* coll_req{};

    friend class ProcessGroupUCC;
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1);

  virtual ~ProcessGroupUCC() override;

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

 protected:
  c10::intrusive_ptr<Store> store_;
  torch_ucx_comm_t* ucx_comm;
  torch_ucc_coll_comm_t* coll_comm;
  torch_ucc_coll_ops_t coll_ops;
  std::mutex pg_mutex;
  std::thread progress_thread;
  bool stop_progress_loop;
  std::list<c10::intrusive_ptr<WorkColl>> progress_list;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;

  void progress_loop();
  c10::intrusive_ptr<ProcessGroup::Work> enqueue_request(
      torch_ucc_coll_request_t* req,
      void* scratch);
  torch_ucc_coll_comm_t* get_coll_comm();

 private:
  struct ucc_config {
    bool enable_progress_thread;
  } config;

  void read_config();
  void check_tensor(const std::vector<at::Tensor>& tensors);
};

} // namespace c10d

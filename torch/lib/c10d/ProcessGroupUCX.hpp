#pragma once

#include <condition_variable>
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
#include <c10d/UCXSendRecv.hpp>
#include <c10d/UCXColl.hpp>

namespace c10d {

class ProcessGroupUCX : public ProcessGroup {
 public:
  class WorkUCX: public ProcessGroup::Work {
   public:
    WorkUCX(torch_ucx_request_t *request, torch_ucx_comm_t *ucx_comm):
        req(request), comm(ucx_comm) {}
    virtual ~WorkUCX() {};
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
   protected:
    torch_ucx_request_t *req;
    torch_ucx_comm_t    *comm;
    friend class ProcessGroupUCX;
  };
  class WorkUCXColl: public ProcessGroup::Work {
   public:
    WorkUCXColl() {
        req = new torch_ucx_coll_request_t;
        no_progress = false;
    }
    virtual ~WorkUCXColl() {};
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
   protected:
    at::Tensor               src;
    at::Tensor               dst;
    bool                     no_progress;
    torch_ucx_coll_request_t *req;
    friend class ProcessGroupUCX;
  };
  explicit ProcessGroupUCX(const std::shared_ptr<Store>& store, int rank, int size, const std::chrono::milliseconds& opTimeout);

  virtual ~ProcessGroupUCX();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag);

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;
 protected:
  torch_ucx_comm_t                      *ucx_comm;
  torch_ucx_coll_comm_t                 *ucx_coll_comm;
  std::shared_ptr<Store>                store_;
  std::mutex                            pg_mutex;
  std::thread                           progress_thread;
  bool                                  stop_progress_loop;
  std::deque<torch_ucx_coll_request_t*> progress_queue;
  std::condition_variable               queue_produce_cv;
  std::condition_variable               queue_consume_cv;
  void progress_loop();
  void enqueue_request(torch_ucx_coll_request_t* req);
  void read_config();
  struct ucc_config {
      bool enable_progress_thread;
      bool enable_xccl;
      bool enable_ucx;
  } config;
};

} // namespace c10d

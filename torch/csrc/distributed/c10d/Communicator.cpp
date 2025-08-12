#include <vector>

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/Communicator.hpp>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/macros/Macros.h>

namespace c10d {

Communicator::Communicator(int rank, int size)
    : rank_(rank), size_(size) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Communicator::~Communicator() = default;

c10::intrusive_ptr<c10d::Work> Communicator::allreduce(std::vector<at::Tensor>& tensors, c10d::ReduceOp reduceOp, bool asyncOp, std::chrono::milliseconds timeout, std::optional<at::Tensor> sparseIndices) {
  static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::allreduce_", "")
                         .typed<
                            std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                                at::TensorList,
                                const c10::intrusive_ptr<::c10d::Communicator>&,
                                const c10::intrusive_ptr<::c10d::ReduceOp>&,
                                const std::optional<at::Tensor>& sparse_indices,
                                bool,
                                int64_t)>();

    auto work = std::get<1>(op.call(
        tensors,
        c10::intrusive_ptr<Communicator>::unsafe_reclaim_from_nonowning(this),
        c10::make_intrusive<ReduceOp>(reduceOp),
        sparseIndices,
        asyncOp,
        timeout.count()));

    if (c10d::allow_inflight_collective_as_graph_input()) {
      for (const auto& tensor : tensors) {
        c10d::register_work(tensor, work);
      }
    }
    return work;
}

c10::intrusive_ptr<c10d::Work> Communicator::allreduce_dispatched(std::vector<at::Tensor>& tensors, c10d::ReduceOp reduceOp, bool asyncOp, std::chrono::milliseconds timeout, std::optional<at::Tensor> sparseIndices) {
  FlightRecorder<c10::Event>::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        "allreduce",
        tensors,
        tensors,
        nullptr,
        nullptr,
        timeout,
        nullptr,
        false);

  auto work = this->allreduceImpl(tensors, reduceOp, asyncOp, timeout, sparseIndices);
  // We need to have a trace_id for the work object.
  FlightRecorder<c10::Event>::get()->retire_id(0, true);
    return work;
}

} // namespace c10d

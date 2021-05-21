#include <c10d/ProcessGroupWrapper.hpp>
#include <stdexcept>
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "c10d/ProcessGroup.hpp"
#include "c10d/ProcessGroupGloo.hpp"

namespace c10d {

// A container for information about a particular collective, including optype
// and input tensors (if applicable.)
struct CollectiveFingerPrint {
  // Current collective's operation type.
  OpType op_type_;
  // Input tensors, if given, of the collective. If given, shapes will be
  // checked across processes to ensure valid input into the collective.
  std::vector<at::Tensor> input_tensors_;
  explicit CollectiveFingerPrint(
      OpType op_type,
      std::vector<at::Tensor> input_tensors)
      : op_type_(op_type), input_tensors_(input_tensors) {}

  // Verifies a given int is the same across processes.
  void verify_num(
      int64_t value,
      const c10::intrusive_ptr<ProcessGroup>& pg,
      const std::string& failureMsg) {
    auto tensor = at::full({1}, value, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> tensors;
    tensors.reserve(pg->getSize());
    for (int i = 0; i < pg->getSize(); ++i) {
      tensors.emplace_back(at::zeros_like(tensor));
    }
    std::vector<std::vector<at::Tensor>> out_tensors({tensors});
    std::vector<at::Tensor> inp_tensors({tensor});
    pg->allgather(out_tensors, inp_tensors)->wait();
    std::unordered_set<int64_t> gathered;
    for (const auto& t : out_tensors[0]) {
      auto n = t.item().to<int64_t>();
      gathered.insert(n);
      if (gathered.size() > 1) {
        TORCH_CHECK(false, failureMsg);
      }
    }
  }

  // Verifies that shapes are consistent across processes.
  // shape_tensors_to_report should be specified as the tensors to report when a
  // shape inconsistency is found. This is not necessarily shape_tensors such as
  // in the case we are checking shape dimensionality.
  void verify_shapes(
      std::vector<at::Tensor> shape_tensors,
      std::vector<at::Tensor> shape_tensors_to_report,
      c10::intrusive_ptr<ProcessGroup>& pg) {
    std::vector<std::vector<at::Tensor>> output_tensors;
    output_tensors.reserve(shape_tensors.size());
    for (int k = 0; k < shape_tensors.size(); ++k) {
      auto& tensor_shape = shape_tensors[k];
      std::vector<at::Tensor> outputs;
      outputs.reserve(pg->getSize());
      for (int i = 0; i < pg->getSize(); ++i) {
        outputs.emplace_back(at::zeros_like(tensor_shape));
      }
      output_tensors.emplace_back(outputs);
    }
    // Allgather tensor shapes.
    pg->allgather(output_tensors, shape_tensors)->wait();
    // Verify equivalence
    for (int i = 0; i < output_tensors.size(); ++i) {
      auto world_tensor_shapes = output_tensors[i];
      auto reference_shape_tensor = shape_tensors[i];
      for (const auto& rank_tensor_shape : world_tensor_shapes) {
        if (!rank_tensor_shape.equal(reference_shape_tensor)) {
          TORCH_CHECK(
              false,
              c10::str(
                  "Error when verifying shape tensors for collective ",
                  opTypeToString(op_type_),
                  " on rank ",
                  pg->getRank(),
                  ". This likely indicates that input shapes into the collective are mismatched across ranks. Got shapes: ",
                  shape_tensors_to_report));
        }
      }
    }
  }

  // Executes and verifies the collective fingerprint.
  void verify(c10::intrusive_ptr<ProcessGroup> pg) {
    // For collectives, all ranks should participate and call into them in the
    // same order. Verify the same operation type is being requested.
    int64_t op_type_int = static_cast<int64_t>(op_type_);
    verify_num(
        op_type_int,
        pg,
        c10::str(
            "Mismatch between collective operation types across ranks.",
            "This likely indicates an application bug where different ranks are ",
            "calling different collectives. ",
            "Rank ",
            pg->getRank(),
            " is calling collective: ",
            opTypeToString(op_type_)));
    // Retrieve input tensor shapes.
    std::vector<at::Tensor> shape_tensors =
        c10d::getTensorShapes(input_tensors_);
    // If input_tensors_ is empty we would get no shape tensors back, but still
    // do verification in case input_tensors_.empty() is
    // inconsistent across ranks. In this case, sub in a single zeros tensor and
    // ensure all ranks agree, because gloo pg does not allow collectives with
    // empty inputs.
    if (shape_tensors.size() == 0) {
      shape_tensors = {at::zeros(1)};
    }
    // Verify dimensionality of shapes. This catches errors where tensor shapes
    // have different dimensions such as torch.randn(2, 3) vs torch.randn(2, 3,
    // 4). If we did not do this step and instead proceeded directly with
    // verifying tensor shapes, we would have malformed input into allgather()
    // and crash with an unhelpful error.
    std::vector<at::Tensor> meta_shape_tensors =
        c10d::getTensorShapes(shape_tensors);

    verify_shapes(
        meta_shape_tensors, /* shape_tensors_to_report= */ shape_tensors, pg);

    // If all meta shapes are 0 then we can skip the below verification since
    // it is not possible that there would be a difference. This happens only
    // when the tensor wraps a single scalar.
    bool skip = true;
    for (int i = 0; i < meta_shape_tensors.size(); ++i) {
      auto& t = meta_shape_tensors[i];
      if (t.item().to<int64_t>() != 0) {
        skip = false;
        break;
      }
    }
    if (!skip) {
      verify_shapes(
          shape_tensors, /* shape_tensors_to_report= */ shape_tensors, pg);
    }
  }
};

ProcessGroupWrapper::ProcessGroupWrapper(
    c10::intrusive_ptr<ProcessGroup> pg,
    c10::intrusive_ptr<ProcessGroupGloo> glooPg)
    : ProcessGroup(pg->getRank(), pg->getSize()), pg_(pg), glooPg_(glooPg) {
  // Set the sequence number for the underlying process group.
  pg_->setSequenceNumberForGroup();
}

const std::string ProcessGroupWrapper::getBackendName() const {
  return pg_->getBackendName();
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::broadcast(
    std::vector<at::Tensor>& data,
    const BroadcastOptions& opts) {
  runCollectiveChecks(OpType::BARRIER, data);
  return pg_->broadcast(data, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::allreduce(
    std::vector<at::Tensor>& data,
    const AllreduceOptions& opts) {
  runCollectiveChecks(OpType::ALLREDUCE, data);
  return pg_->allreduce(data, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  // NOTE: We don't enforce shape checking for allreduce_coalesced because
  // the implementation itself does not enforce it we have tests that use
  // inconsistent shapes, see python implementation in distributed_c10d for
  // details.
  runCollectiveChecks(OpType::ALLREDUCE_COALESCED, {});
  return pg_->allreduce_coalesced(tensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  runCollectiveChecks(OpType::REDUCE, tensors);
  return pg_->reduce(tensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  runCollectiveChecks(OpType::ALLGATHER, inputTensors);
  return pg_->allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  std::vector<at::Tensor> inputTensors({inputBuffer});
  runCollectiveChecks(OpType::_ALLGATHER_BASE, inputTensors);
  return pg_->_allgather_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  // NOTE: We don't enforce shape checking for allgather_coalesced because
  // the implementation itself does not enforce it we have tests that use
  // inconsistent shapes, see python implementation in distributed_c10d for
  // details.
  runCollectiveChecks(OpType::ALLGATHER_COALESCED, {});
  return pg_->allgather_coalesced(outputTensorLists, inputTensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  runCollectiveChecks(OpType::GATHER, inputTensors);
  return pg_->gather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  runCollectiveChecks(OpType::SCATTER, outputTensors);
  return pg_->scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  runCollectiveChecks(OpType::REDUCE_SCATTER, outputTensors);
  return pg_->reduce_scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  // alltoall supports uneven split, so don't enforce shape checking.
  runCollectiveChecks(OpType::ALLTOALL_BASE, {});
  return pg_->alltoall_base(
      outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  // alltoall supports uneven split, so don't enforce shape checking.
  runCollectiveChecks(OpType::ALLTOALL, {});
  return pg_->alltoall(outputTensors, inputTensors, opts);
}

void ProcessGroupWrapper::monitoredBarrier(
    const BarrierOptions& opts,
    bool waitAllRanks) {
  return pg_->monitoredBarrier(opts, waitAllRanks);
}

void ProcessGroupWrapper::setSequenceNumberForGroup() {
  // Set underlying pg's sequence number if it is not set.
  if (pg_->getSequenceNumberForGroup() == 0) {
    // Set the sequence number for the underlying process group.
    pg_->setSequenceNumberForGroup();
  }
}

uint64_t ProcessGroupWrapper::getSequenceNumberForGroup() {
  return pg_->getSequenceNumberForGroup();
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  return pg_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  return pg_->recv(tensors, srcRank, tag);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  return pg_->recvAnysource(tensors, tag);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupWrapper::barrier(
    const BarrierOptions& opts) {
  runCollectiveChecks(OpType::BARRIER, {});
  return pg_->barrier(opts);
}

void ProcessGroupWrapper::runCollectiveChecks(
    OpType op_type,
    const std::vector<at::Tensor>& tensors) const {
  // first perform a monitored barrier to ensure all ranks can synchronize.
  c10d::BarrierOptions options;
  // TODO: we should use wrapped pg_'s timeout here, but C++ ProcessGroup API
  // does not expose timeout.
  glooPg_->monitoredBarrier(options, /* waitAllRanks */ true);
  auto finger_print = CollectiveFingerPrint(op_type, tensors);
  // Will throw if an ill-formed collective is detected.
  finger_print.verify(glooPg_);
}

} // namespace c10d

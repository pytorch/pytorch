#include <c10d/ProcessGroupWrapper.hpp>

#ifdef USE_C10D_GLOO

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <stdexcept>

namespace c10d {

namespace {
// A container for information about a particular collective, including optype
// and input tensors (if applicable.)
struct CollectiveFingerPrint {
  // Current collective's operation type.
  OpType op_type_;
  // Ref to input tensors, if given, of the collective. If given, shapes will be
  // checked across processes to ensure valid input into the collective.
  const std::vector<at::Tensor>& input_tensors_;
  // input tensor data types
  std::vector<int8_t> tensor_dtypes_;
  // input tensor device types
  std::vector<int8_t> tensor_device_types_;
  explicit CollectiveFingerPrint(
      OpType op_type,
      const std::vector<at::Tensor>& input_tensors)
      : op_type_(op_type), input_tensors_(input_tensors) {
    tensor_dtypes_.reserve(input_tensors.size());
    tensor_device_types_.reserve(input_tensors.size());
    for (const at::Tensor& t : input_tensors_) {
      tensor_dtypes_.push_back(static_cast<int8_t>(t.dtype().toScalarType()));
      tensor_device_types_.push_back(static_cast<int8_t>(t.device().type()));
    }
  }

  // Logs collective information in case of a failure.
  friend std::ostream& operator<<(
      std::ostream& output,
      const CollectiveFingerPrint& collective_fingerprint);

  at::Tensor serialize_fingerprint() {
    auto data = std::make_unique<std::vector<int64_t>>();
    // std::vector<int64_t> data;
    // OpType
    data->push_back(static_cast<int64_t>(op_type_));
    // Shapes
    for (const auto& tensor : input_tensors_) {
      auto sizes = tensor.sizes().vec();
      for (const auto& s : sizes) {
        data->push_back(s);
      }
    }
    // tensor dtypes
    for (const auto& type : tensor_dtypes_) {
      data->push_back(type);
    }
    // device types
    for (const auto& d : tensor_device_types_) {
      data->push_back(d);
    }
    // Serialize data into tensor
    int64_t data_size = data->size();
    // Need to release here and get the ptr due to C++ parameter evaluation
    // order.
    auto d = data.release();
    at::Tensor serialized_tensor =
        at::for_blob(d->data(), {data_size})
            .context(
                d,
                [](void* ctx) {
                  delete static_cast<std::vector<int64_t>*>(ctx);
                })
            .options(at::TensorOptions().dtype(at::kLong))
            .make_tensor();
    return serialized_tensor;
  }

  void verify_tensors(
      std::vector<at::Tensor>& tensors_to_verify,
      c10::intrusive_ptr<ProcessGroup>& pg) {
    // Create output tensor data structure to pass into allgather.
    std::vector<std::vector<at::Tensor>> output_tensors;
    output_tensors.reserve(tensors_to_verify.size());
    for (auto& tensor_shape : tensors_to_verify) {
      std::vector<at::Tensor> outputs;
      outputs.reserve(pg->getSize());
      for (int i = 0; i < pg->getSize(); ++i) {
        outputs.emplace_back(at::zeros_like(tensor_shape));
      }
      output_tensors.emplace_back(outputs);
    }
    // Allgather tensor shapes.
    pg->allgather(output_tensors, tensors_to_verify)->wait();
    // Verify equivalence
    for (const auto i : c10::irange(output_tensors.size())) {
      const std::vector<at::Tensor> gathered_tensors = output_tensors[i];
      const at::Tensor reference_tensor = tensors_to_verify[i];
      for (const auto& rank_tensor : gathered_tensors) {
        if (!rank_tensor.equal(reference_tensor)) {
          std::stringstream ss;
          ss << "Detected mismatch between collectives on ranks. Rank "
             << pg->getRank()
             << " is running inconsistent collective: " << *this;
          TORCH_CHECK(false, ss.str());
        }
      }
    }
  }

  // Executes and verifies the collective fingerprint.
  void verify(c10::intrusive_ptr<ProcessGroup> pg) {
    at::Tensor serialized_tensor = serialize_fingerprint();
    std::vector<at::Tensor> inp{serialized_tensor};
    // First verify tensor shapes. This is needed because if e.g. tensor dim
    // does not match across processes, directly verifying tensors will result
    // in a crash during allgather, but we'd actually like to report a
    // description about the inconsistency. Since the input is just a 1D tensor
    // the shape will be a single int k_i and we need to make sure k_i is
    // consistent across the whole world.
    std::vector<at::Tensor> sp = c10d::getTensorShapes(inp);
    verify_tensors(sp, pg);
    // Now verify consistency for the actual tensor.
    verify_tensors(inp, pg);
  }
};

std::ostream& operator<<(
    std::ostream& output,
    const CollectiveFingerPrint& collective_fingerprint) {
  std::string collectiveInfo;
  if (!collective_fingerprint.input_tensors_.empty()) {
    // Convert dtype and device type info to string.
    std::vector<std::string> dtype_strs;
    std::vector<std::string> device_type_strs;
    for (const auto& tensor_dtype : collective_fingerprint.tensor_dtypes_) {
      dtype_strs.push_back(
          c10::toString(static_cast<at::ScalarType>(tensor_dtype)));
    }
    for (const auto& tensor_device_type :
         collective_fingerprint.tensor_device_types_) {
      device_type_strs.push_back(
          c10::toString(static_cast<at::DeviceType>(tensor_device_type)));
    }

    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "OpType=",
        opTypeToString(collective_fingerprint.op_type_),
        ", TensorShape=",
        (collective_fingerprint.input_tensors_)[0].sizes(),
        ", TensorDtypes=",
        (dtype_strs),
        ", TensorDeviceTypes=",
        (device_type_strs));
  } else {
    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "OpType=",
        opTypeToString(collective_fingerprint.op_type_));
  }
  return output << collectiveInfo;
}

} // namespace

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
  runCollectiveChecks(OpType::BROADCAST, data);
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

#endif // USE_C10D_GLOO

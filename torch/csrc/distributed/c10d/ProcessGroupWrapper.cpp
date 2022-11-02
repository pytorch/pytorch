#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>

#ifdef USE_C10D_GLOO

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <stdexcept>

namespace c10d {

namespace {
// A container for information about a particular collective, including optype
// and input tensors (if applicable.)
struct CollectiveFingerPrint {
  // Current collective's operation type.
  OpType op_type_;
  // Number of input tensors
  std::size_t num_tensors_;
  // input tensor data types
  std::vector<int8_t> tensor_dtypes_;
  // input tensor device types
  std::vector<int8_t> tensor_device_types_;
  // input tensor sizes
  std::vector<std::vector<int64_t>> tensor_sizes_;

  explicit CollectiveFingerPrint(
      OpType op_type,
      const std::vector<at::Tensor>& input_tensors)
      : op_type_(op_type), num_tensors_(input_tensors.size()) {
    tensor_dtypes_.reserve(num_tensors_);
    tensor_device_types_.reserve(num_tensors_);
    tensor_sizes_.reserve(num_tensors_);
    for (const at::Tensor& t : input_tensors) {
      tensor_dtypes_.push_back(static_cast<int8_t>(t.dtype().toScalarType()));
      tensor_device_types_.push_back(static_cast<int8_t>(t.device().type()));
      tensor_sizes_.push_back(t.sizes().vec());
    }
  }

  // Constructor for the data received from deserialized fingerprint
  CollectiveFingerPrint(
      OpType op_type,
      std::vector<int8_t> tensor_dtypes,
      std::vector<int8_t> tensor_device_types,
      std::vector<std::vector<int64_t>> tensor_sizes)
      : op_type_(op_type),
        tensor_dtypes_(tensor_dtypes),
        tensor_device_types_(tensor_device_types),
        tensor_sizes_(tensor_sizes) {}

  // Logs collective information in case of a failure.
  friend std::ostream& operator<<(
      std::ostream& output,
      const CollectiveFingerPrint& collective_fingerprint);

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

  // Takes a serialized fingerprint from
  // CollectiveFingerPrint::serialize_fingerprint and deserializes it back to a
  // CollectiveFingerPrint struct
  CollectiveFingerPrint deserialize_fingerprint(at::Tensor serialized_tensor) {
    OpType optype;
    auto dtypes = std::vector<int8_t>();
    auto device_types = std::vector<int8_t>();
    auto sizes = std::vector<std::vector<int64_t>>();
    int index = 0;
    // 1. OpType
    optype = OpType(serialized_tensor[index].item<int>());
    index++;

    if (index < serialized_tensor.size(0)) {
      // 2. Num tensors
      int num_tensors = serialized_tensor[index].item<int>();
      index++;
      dtypes.reserve(num_tensors);
      device_types.reserve(num_tensors);
      sizes.reserve(num_tensors);

      // 3. Tensor dtypes
      for (int i = 0; i < num_tensors; i++) {
        dtypes.push_back(serialized_tensor[index].item<int8_t>());
        index++;
      }
      // 4. Device types
      for (int i = 0; i < num_tensors; i++) {
        device_types.push_back(serialized_tensor[index].item<int8_t>());
        index++;
      }
      // 5. Tensor shapes
      for (int i = 0; i < num_tensors; i++) {
        // 5a. Shape size
        int size = serialized_tensor[index].item<int>();
        index++;
        // 5b. Shape
        auto shapeVec = std::vector<int64_t>();
        shapeVec.reserve(size);
        for (int j = 0; j < size; j++) {
          shapeVec.push_back(serialized_tensor[index].item<int64_t>());
          index++;
        }
        sizes.push_back(shapeVec);
      }
    }
    return CollectiveFingerPrint(optype, dtypes, device_types, sizes);
  }

 private:
  void verify_tensors(
      std::vector<at::Tensor>& tensors_to_verify,
      c10::intrusive_ptr<ProcessGroup>& pg) {
    // Create output tensor data structure to pass into allgather.
    std::vector<std::vector<at::Tensor>> output_tensors;
    // output tensors: [<tensor 0 outputs>, <tensor 1 outputs>, ..., <tensor n
    // outputs>]
    output_tensors.reserve(tensors_to_verify.size());
    for (const auto& tensor_shape : tensors_to_verify) {
      // Each rank has its own outputs shape, e.g.
      // <tensor 0 outputs>: [<rank 0 tensor>, <rank 1 tensor>, ..., <rank n
      // tensor>]
      std::vector<at::Tensor> outputs;
      outputs.reserve(pg->getSize());
      for (const auto i : c10::irange(pg->getSize())) {
        std::ignore = i; // Suppress unused variable warning
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
      for (int rank = 0; rank < gathered_tensors.size(); rank++) {
        const auto& rank_tensor = gathered_tensors[rank];
        if (!rank_tensor.equal(reference_tensor)) {
          CollectiveFingerPrint rank_fingerprint =
              deserialize_fingerprint(rank_tensor);
          std::stringstream ss;
          ss << "Detected mismatch between collectives on ranks. Rank "
             << pg->getRank() << " is running collective: " << *this
             << ", but Rank " << rank
             << " is running collective: " << rank_fingerprint << ".";
          TORCH_CHECK(false, ss.str());
        }
      }
    }
  }

  // Serializes the information (op type, input shapes, data types, device
  // types) about the collective fingerprint into a tensor
  at::Tensor serialize_fingerprint() {
    auto data = std::make_unique<std::vector<int64_t>>();
    // std::vector<int64_t> data;
    // 1. OpType
    data->push_back(static_cast<int64_t>(op_type_));
    // 2. Num tensors
    data->push_back(static_cast<int64_t>(num_tensors_));
    // 3. Tensor dtypes
    for (const auto& type : tensor_dtypes_) {
      data->push_back(type);
    }
    // 4. Device types
    for (const auto& d : tensor_device_types_) {
      data->push_back(d);
    }
    // 5. Shapes
    for (const auto& sizes : tensor_sizes_) {
      data->push_back(sizes.size());
      for (const auto& s : sizes) {
        data->push_back(s);
      }
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
};

std::ostream& operator<<(
    std::ostream& output,
    const CollectiveFingerPrint& collective_fingerprint) {
  std::string collectiveInfo;
  if (collective_fingerprint.num_tensors_ != 0) {
    // Convert dtype and device type info to string.
    std::vector<std::string> dtype_strs;
    std::vector<std::string> device_type_strs;
    std::vector<std::string> size_strs;
    for (const auto& tensor_dtype : collective_fingerprint.tensor_dtypes_) {
      dtype_strs.emplace_back(
          c10::toString(static_cast<at::ScalarType>(tensor_dtype)));
    }
    for (const auto& tensor_device_type :
         collective_fingerprint.tensor_device_types_) {
      device_type_strs.emplace_back(
          c10::toString(static_cast<at::DeviceType>(tensor_device_type)));
    }
    if (!collective_fingerprint.tensor_sizes_.empty()) {
      for (const auto& single_tensor_shape_num :
           collective_fingerprint.tensor_sizes_[0]) {
        size_strs.emplace_back(std::to_string(single_tensor_shape_num));
      }
    }

    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "OpType=",
        opTypeToString(collective_fingerprint.op_type_),
        ", TensorShape=[",
        c10::Join(", ", size_strs),
        "], TensorDtypes=",
        (dtype_strs),
        ", TensorDeviceTypes=",
        (device_type_strs),
        ")");
  } else {
    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "OpType=",
        opTypeToString(collective_fingerprint.op_type_),
        ")");
  }
  return output << collectiveInfo;
}

} // namespace

ProcessGroupWrapper::ProcessGroupWrapper(
    c10::intrusive_ptr<ProcessGroup> pg,
    c10::intrusive_ptr<ProcessGroup> glooPg)
    : Backend(pg->getRank(), pg->getSize()), pg_(pg), glooPg_(glooPg) {
  // Set the sequence number for the underlying process group.
  pg_->setSequenceNumberForGroup();
}

const std::string ProcessGroupWrapper::getBackendName() const {
  return pg_->getBackendName();
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::broadcast(
    std::vector<at::Tensor>& data,
    const BroadcastOptions& opts) {
  runCollectiveChecks(OpType::BROADCAST, data);
  return pg_->broadcast(data, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allreduce(
    std::vector<at::Tensor>& data,
    const AllreduceOptions& opts) {
  runCollectiveChecks(OpType::ALLREDUCE, data);
  return pg_->allreduce(data, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  // NOTE: We don't enforce shape checking for allreduce_coalesced because
  // the implementation itself does not enforce it we have tests that use
  // inconsistent shapes, see python implementation in distributed_c10d for
  // details.
  runCollectiveChecks(OpType::ALLREDUCE_COALESCED, {});
  return pg_->allreduce_coalesced(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  runCollectiveChecks(OpType::REDUCE, tensors);
  return pg_->reduce(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  runCollectiveChecks(OpType::ALLGATHER, inputTensors);
  return pg_->allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  std::vector<at::Tensor> inputTensors({inputBuffer});
  runCollectiveChecks(OpType::_ALLGATHER_BASE, inputTensors);
  return pg_->_allgather_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allgather_coalesced(
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

c10::intrusive_ptr<Work> ProcessGroupWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  runCollectiveChecks(OpType::GATHER, inputTensors);
  return pg_->gather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  runCollectiveChecks(OpType::SCATTER, outputTensors);
  return pg_->scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  runCollectiveChecks(OpType::REDUCE_SCATTER, outputTensors);
  return pg_->reduce_scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::alltoall_base(
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

c10::intrusive_ptr<Work> ProcessGroupWrapper::alltoall(
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

c10::intrusive_ptr<Work> ProcessGroupWrapper::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  return pg_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  return pg_->recv(tensors, srcRank, tag);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  return pg_->recvAnysource(tensors, tag);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::barrier(
    const BarrierOptions& opts) {
  runCollectiveChecks(OpType::BARRIER, {});
  return pg_->barrier(opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::_reduce_scatter_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ReduceScatterOptions& opts) {
  runCollectiveChecks(
      OpType::_REDUCE_SCATTER_BASE, {inputBuffer, outputBuffer});
  return pg_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupWrapper::getWrappedPg() const {
  return pg_;
}

void ProcessGroupWrapper::runCollectiveChecks(
    OpType op_type,
    const std::vector<at::Tensor>& tensors) const {
  // first perform a monitored barrier to ensure all ranks can synchronize.
  c10d::BarrierOptions options;
  // TODO: we should use wrapped pg_'s timeout here, but C++ ProcessGroup API
  // does not expose timeout.
  auto finger_print = CollectiveFingerPrint(op_type, tensors);
  try {
    glooPg_->monitoredBarrier(options, /* waitAllRanks */ true);
  } catch (const std::runtime_error& e) {
    // Attach collective info to the exception and re-raise.
    std::stringstream ss;
    ss << finger_print;
    auto collective_info = ss.str();
    auto err_msg = c10::str(
        "ProcessGroupWrapper: Monitored Barrier encountered error running collective: ",
        collective_info,
        ". Error: \n",
        e.what());
    TORCH_CHECK(false, err_msg);
  }
  // Will throw if an ill-formed collective is detected.
  finger_print.verify(glooPg_);
}

} // namespace c10d

#endif // USE_C10D_GLOO

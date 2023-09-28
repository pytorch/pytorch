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
#include <utility>

namespace c10d {

namespace {
// A container for information about a particular collective, including optype
// and input tensors (if applicable.)
struct CollectiveFingerPrint {
  // Current collective's operation type.
  OpType op_type_;
  // Number of input tensors
  std::size_t num_tensors_{};
  // input tensor data types
  std::vector<int8_t> tensor_dtypes_;
  // input tensor device types
  std::vector<int8_t> tensor_device_types_;
  // input tensor sizes
  std::vector<std::vector<int64_t>> tensor_sizes_;
  int sequence_number_;

  CollectiveFingerPrint(
      OpType op_type,
      const std::vector<at::Tensor>& input_tensors,
      int sequence_number)
      : op_type_(op_type),
        num_tensors_(input_tensors.size()),
        sequence_number_(sequence_number) {
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
      size_t num_tensors,
      std::vector<int8_t> tensor_dtypes,
      std::vector<int8_t> tensor_device_types,
      std::vector<std::vector<int64_t>> tensor_sizes,
      int sequence_number)
      : op_type_(op_type),
        num_tensors_(num_tensors),
        tensor_dtypes_(std::move(tensor_dtypes)),
        tensor_device_types_(std::move(tensor_device_types)),
        tensor_sizes_(std::move(tensor_sizes)),
        sequence_number_(sequence_number) {}

  // Logs collective information in case of a failure.
  friend std::ostream& operator<<(
      std::ostream& output,
      const CollectiveFingerPrint& collective_fingerprint);

  // Executes and verifies the collective fingerprint.
  void verify(c10::intrusive_ptr<Backend> backend) {
    at::Tensor serialized_tensor = serialize_fingerprint();
    std::vector<at::Tensor> inp{serialized_tensor};
    // First verify tensor shapes. This is needed because if e.g. tensor dim
    // does not match across processes, directly verifying tensors will result
    // in a crash during allgather, but we'd actually like to report a
    // description about the inconsistency. Since the input is just a 1D tensor
    // the shape will be a single int k_i and we need to make sure k_i is
    // consistent across the whole world.
    std::vector<at::Tensor> sp = c10d::getTensorShapes(inp);
    verify_tensors(sp, backend);
    // Now verify consistency for the actual tensor.
    verify_tensors(inp, backend);
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
    int seq = 0;
    // 1. OpType
    optype = OpType(serialized_tensor[index].item<int>());
    index++;
    int num_tensors = 0;
    if (index < serialized_tensor.size(0)) {
      seq = serialized_tensor[index].item<int64_t>();
      index++;
      // 2. Num tensors
      num_tensors = serialized_tensor[index].item<int>();
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
    return CollectiveFingerPrint(
        optype, num_tensors, dtypes, device_types, sizes, seq);
  }

 private:
  void verify_tensors(
      std::vector<at::Tensor>& tensors_to_verify,
      c10::intrusive_ptr<Backend>& backend) {
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
      outputs.reserve(backend->getSize());
      for (const auto i : c10::irange(backend->getSize())) {
        std::ignore = i; // Suppress unused variable warning
        outputs.emplace_back(at::zeros_like(tensor_shape));
      }
      output_tensors.emplace_back(outputs);
    }
    // Allgather tensor shapes.
    backend->allgather(output_tensors, tensors_to_verify)->wait();
    // Verify equivalence
    for (const auto i : c10::irange(output_tensors.size())) {
      const std::vector<at::Tensor> gathered_tensors = output_tensors[i];
      const at::Tensor reference_tensor = tensors_to_verify[i];
      for (const auto rank : c10::irange(gathered_tensors.size())) {
        const auto& rank_tensor = gathered_tensors[rank];
        if (!rank_tensor.equal(reference_tensor)) {
          CollectiveFingerPrint rank_fingerprint =
              deserialize_fingerprint(rank_tensor);
          std::stringstream ss;
          ss << "Detected mismatch between collectives on ranks. Rank "
             << backend->getRank() << " is running collective: " << *this
             << ", but Rank " << rank
             << " is running collective: " << rank_fingerprint << ".";
          auto diff_result = compute_collective_diff(rank_fingerprint);
          if (std::get<0>(diff_result)) {
            ss << std::get<1>(diff_result);
          }

          TORCH_CHECK(false, ss.str());
        }
      }
    }
  }

  static std::vector<std::string> get_size_strs(
      const CollectiveFingerPrint& collective_fingerprint) {
    std::vector<std::string> size_strs;
    if (!collective_fingerprint.tensor_sizes_.empty()) {
      for (const auto& single_tensor_shape_num :
           collective_fingerprint.tensor_sizes_[0]) {
        size_strs.emplace_back(std::to_string(single_tensor_shape_num));
      }
    }
    return size_strs;
  }

  static std::vector<std::string> get_dtype_strs(
      const CollectiveFingerPrint& collective_fingerprint) {
    std::vector<std::string> dtype_strs;
    dtype_strs.reserve(collective_fingerprint.tensor_dtypes_.size());
    for (const auto& tensor_dtype : collective_fingerprint.tensor_dtypes_) {
      dtype_strs.emplace_back(
          c10::toString(static_cast<at::ScalarType>(tensor_dtype)));
    }
    return dtype_strs;
  }

  static std::vector<std::string> get_device_type_strs(
      const CollectiveFingerPrint& collective_fingerprint) {
    std::vector<std::string> device_type_strs;
    device_type_strs.reserve(
        collective_fingerprint.tensor_device_types_.size());
    for (const auto& tensor_device_type :
         collective_fingerprint.tensor_device_types_) {
      device_type_strs.emplace_back(
          c10::toString(static_cast<at::DeviceType>(tensor_device_type)));
    }
    return device_type_strs;
  }

  std::pair<bool, std::string> compute_collective_diff(
      CollectiveFingerPrint& other) {
    // Computes the difference between two collectives (seq num, tensor shapes,
    // collective type, etc) for easier understanding of how mismatched
    // collectives across ranks differ.
    bool found_diff = false;
    std::stringstream ss;
    ss << "Collectives differ in the following aspects: ";
    // Check seq_num
    if (other.sequence_number_ != sequence_number_) {
      found_diff = true;
      ss << c10::str(
          "\t Sequence number: ",
          sequence_number_,
          "vs ",
          other.sequence_number_);
    }
    // Check op type
    auto other_op = opTypeToString(other.op_type_);
    auto this_op = opTypeToString(op_type_);
    if (other_op.compare(this_op) != 0) {
      found_diff = true;
      ss << c10::str("  Op type: ", this_op, "vs ", other_op);
    }

    auto check = [&ss, &found_diff](
                     const char* arg,
                     std::vector<std::string> other,
                     std::vector<std::string> curr) {
      if (other.size() != curr.size()) {
        found_diff = true;
        ss << c10::str("  Tensor ", arg, ": ", curr, "vs ", other);
        return;
      }
      for (size_t i = 0; i < other.size(); ++i) {
        if (other[i].compare(curr[i]) != 0) {
          found_diff = true;
          ss << c10::str("  Tensor ", arg, ": ", curr, "vs ", other);
          return;
        }
      }
    };

    // check tensor sizes
    auto other_sizes = get_size_strs(other);
    auto this_sizes = get_size_strs(*this);
    check("Tensor shapes", other_sizes, this_sizes);

    // check tensor dtypes
    auto other_dtypes = get_dtype_strs(other);
    auto this_dtypes = get_dtype_strs(*this);
    check("Tensor dtypes", other_dtypes, this_dtypes);

    // check tensor devices
    auto other_devices = get_device_type_strs(other);
    auto this_devices = get_device_type_strs(*this);

    check("Tensor devices", other_devices, this_devices);
    if (!found_diff) {
      return std::make_pair(false, ss.str());
    } else {
      return std::make_pair(true, ss.str());
    }
  }

  // Serializes the information (op type, input shapes, data types, device
  // types) about the collective fingerprint into a tensor
  at::Tensor serialize_fingerprint() {
    auto data = std::make_unique<std::vector<int64_t>>();
    // std::vector<int64_t> data;
    // 1. OpType
    data->push_back(static_cast<int64_t>(op_type_));
    // sequence number
    data->push_back(sequence_number_);
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
  auto op_type_str = opTypeToString(collective_fingerprint.op_type_);
  if (collective_fingerprint.num_tensors_ != 0) {
    // Convert dtype and device type info to string.
    std::vector<std::string> dtype_strs =
        CollectiveFingerPrint::get_dtype_strs(collective_fingerprint);
    std::vector<std::string> device_type_strs =
        CollectiveFingerPrint::get_device_type_strs(collective_fingerprint);
    std::vector<std::string> size_strs =
        CollectiveFingerPrint::get_size_strs(collective_fingerprint);

    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "SequenceNumber=",
        collective_fingerprint.sequence_number_,
        ", OpType=",
        op_type_str,
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
        "SequenceNumber=",
        collective_fingerprint.sequence_number_,
        "OpType=",
        op_type_str,
        ")");
  }
  return output << collectiveInfo;
}

} // namespace

ProcessGroupWrapper::ProcessGroupWrapper(
    c10::intrusive_ptr<Backend> backend,
    c10::intrusive_ptr<Backend> glooBackend)
    : Backend(backend->getRank(), backend->getSize()),
      backend_(backend),
      glooBackend_(std::move(glooBackend)) {
  // Set the sequence number for the underlying process group.
  backend_->setSequenceNumberForGroup();
}

const std::string ProcessGroupWrapper::getBackendName() const {
  return backend_->getBackendName();
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::broadcast(
    std::vector<at::Tensor>& data,
    const BroadcastOptions& opts) {
  runCollectiveChecks(OpType::BROADCAST, data);
  return backend_->broadcast(data, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allreduce(
    std::vector<at::Tensor>& data,
    const AllreduceOptions& opts) {
  runCollectiveChecks(OpType::ALLREDUCE, data);
  return backend_->allreduce(data, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  // NOTE: We don't enforce shape checking for allreduce_coalesced because
  // the implementation itself does not enforce it we have tests that use
  // inconsistent shapes, see python implementation in distributed_c10d for
  // details.
  runCollectiveChecks(OpType::ALLREDUCE_COALESCED, {});
  return backend_->allreduce_coalesced(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  runCollectiveChecks(OpType::REDUCE, tensors);
  return backend_->reduce(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  runCollectiveChecks(OpType::ALLGATHER, inputTensors);
  return backend_->allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  std::vector<at::Tensor> inputTensors({inputBuffer});
  runCollectiveChecks(OpType::_ALLGATHER_BASE, inputTensors);
  return backend_->_allgather_base(outputBuffer, inputBuffer, opts);
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
  return backend_->allgather_coalesced(outputTensorLists, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  runCollectiveChecks(OpType::GATHER, inputTensors);
  return backend_->gather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  runCollectiveChecks(OpType::SCATTER, outputTensors);
  return backend_->scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  runCollectiveChecks(OpType::REDUCE_SCATTER, outputTensors);
  return backend_->reduce_scatter(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  // alltoall supports uneven split, so don't enforce shape checking.
  runCollectiveChecks(OpType::ALLTOALL_BASE, {});
  return backend_->alltoall_base(
      outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  // alltoall supports uneven split, so don't enforce shape checking.
  runCollectiveChecks(OpType::ALLTOALL, {});
  return backend_->alltoall(outputTensors, inputTensors, opts);
}

void ProcessGroupWrapper::monitoredBarrier(
    const BarrierOptions& opts,
    bool waitAllRanks) {
  return backend_->monitoredBarrier(opts, waitAllRanks);
}

void ProcessGroupWrapper::setSequenceNumberForGroup() {
  // Set underlying pg's sequence number if it is not set.
  if (backend_->getSequenceNumberForGroup() == 0) {
    // Set the sequence number for the underlying process group.
    backend_->setSequenceNumberForGroup();
  }
}

uint64_t ProcessGroupWrapper::getSequenceNumberForGroup() {
  return backend_->getSequenceNumberForGroup();
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  return backend_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  return backend_->recv(tensors, srcRank, tag);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  return backend_->recvAnysource(tensors, tag);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::barrier(
    const BarrierOptions& opts) {
  runCollectiveChecks(OpType::BARRIER, {});
  return backend_->barrier(opts);
}

c10::intrusive_ptr<Work> ProcessGroupWrapper::_reduce_scatter_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ReduceScatterOptions& opts) {
  runCollectiveChecks(
      OpType::_REDUCE_SCATTER_BASE, {inputBuffer, outputBuffer});
  return backend_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<Backend> ProcessGroupWrapper::getWrappedPg() const {
  return backend_;
}

void ProcessGroupWrapper::runCollectiveChecks(
    OpType op_type,
    const std::vector<at::Tensor>& tensors) {
  // first perform a monitored barrier to ensure all ranks can synchronize.
  c10d::BarrierOptions options;
  // TODO: we should use wrapped backend_'s timeout here, but C++ ProcessGroup
  // API does not expose timeout.
  auto seq = getSequenceNumberForGroup();
  auto finger_print = CollectiveFingerPrint(op_type, tensors, seq);
  LOG(INFO) << "[Rank " << getRank() << "] "
            << "Running collective: " << finger_print;
  try {
    glooBackend_->monitoredBarrier(options, /* waitAllRanks */ true);
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
  finger_print.verify(glooBackend_);
}

} // namespace c10d

#endif // USE_C10D_GLOO

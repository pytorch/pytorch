#include <torch/csrc/distributed/c10d/comm.hpp>

#include <deque>

#include <ATen/core/functional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/reducer.hpp>
#include <torch/csrc/utils/tensor_flatten.h>

namespace c10d {
namespace {

class BroadcastWork {
 public:
  BroadcastWork(
      const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
      std::vector<at::Tensor> bucket_tensors,
      int root_rank = 0)
      : bucket_tensors_(std::move(bucket_tensors)),
        flat_tensor_({torch::utils::flatten_dense_tensors(bucket_tensors_)}) {
    BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = root_rank;
    work_ = process_group->broadcast(flat_tensor_, broadcastOptions);
  }

  void finish() {
    work_->wait();

    // Copy the output of the broadcast operation back.
    auto output_tensors = torch::utils::unflatten_dense_tensors(
        flat_tensor_.front(), bucket_tensors_);
    TORCH_INTERNAL_ASSERT(output_tensors.size() == bucket_tensors_.size());
    for (const auto i : c10::irange(output_tensors.size())) {
      // if output_tensor is empty, no need to copy it back,
      // this can avoid error when both bucket_tensor and output_tensor
      // are empty, but they have different shapes, see
      // https://github.com/pytorch/pytorch/issues/87280
      if (output_tensors[i].numel() != 0) {
        bucket_tensors_[i].copy_(output_tensors[i], /*non_blocking=*/true);
      }
    }
  }

 protected:
  // The list of tensors to broadcast. They are guaranteed to be
  // placed on the same device and have the same dtype.
  std::vector<at::Tensor> bucket_tensors_;

  // The vector with a single flattened tensor containing the contents
  // of the tensors in bucket_tensors_. It must be stored in a vector
  // because c10d::ProcessGroup::broadcast takes a vector argument.
  std::vector<at::Tensor> flat_tensor_;

 private:
  // The broadcast work that is kicked off upon construction.
  c10::intrusive_ptr<c10d::Work> work_;
};

} // namespace

// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank) {
  // Coalesce tensors into buckets taking into account the maximum buffer size.
  // This routine is multi-device aware, so the tensors can be split across
  // multiple devices and can contain a mix of CPU and CUDA tensors.
  std::vector<std::vector<size_t>> buckets;
  std::tie(buckets, std::ignore) =
      compute_bucket_assignment_by_size(tensors.vec(), {buffer_size});

  // Returns tensor at specified index in input tensor list.
  const auto lookup = [&tensors](size_t index) { return tensors[index]; };

  // We maintain a maximum of 2 in flight broadcast operations to avoid
  // allocating too much memory (in case the specified tensors are very large).
  std::deque<BroadcastWork> in_flight;
  constexpr auto max_in_flight = 2;
  for (const auto& bucket : buckets) {
    if (in_flight.size() >= max_in_flight) {
      in_flight.front().finish();
      in_flight.pop_front();
    }

    in_flight.emplace_back(process_group, c10::fmap(bucket, lookup), rank);
  }

  while (!in_flight.empty()) {
    in_flight.front().finish();
    in_flight.pop_front();
  }
}

std::vector<at::Tensor> GradBucket::getGradients() const {
  std::vector<at::Tensor> per_parameter_tensors;
  size_t num_parameters = offsets_.size();
  per_parameter_tensors.reserve(num_parameters);
  for (const auto i : c10::irange(num_parameters)) {
    per_parameter_tensors.push_back(
        buffer_.slice(0, offsets_[i], offsets_[i] + lengths_[i])
            .view(sizes_vec_[i]));
  }
  return per_parameter_tensors;
}
namespace detail {

at::Tensor parseCppCommHookResult(const c10::IValue& result) {
  if (result.isPyObject()) {
    std::vector<at::Tensor> tensors =
        result.toPyObjectHolder()->extractTensors();
    return tensors[0];
  }
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList found ",
      result.tagKind());

  if (result.isTensor()) {
    return result.toTensor();
  }

  return result.toTensorVector()[0];
}

} // namespace detail

} // namespace c10d

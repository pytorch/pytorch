#include <torch/csrc/distributed/c10d/comm.h>

#include <deque>

#include <ATen/core/functional.h>
#include <torch/csrc/distributed/c10d/reducer.h>
#include <torch/csrc/utils/tensor_flatten.h>

namespace c10d {
namespace {

class BroadcastWork {
 public:
  BroadcastWork(
      const std::shared_ptr<c10d::ProcessGroup>& process_group,
      std::vector<at::Tensor> bucket_tensors)
      : bucket_tensors_(std::move(bucket_tensors)),
        flat_tensor_({torch::utils::flatten_dense_tensors(bucket_tensors_)}),
        work_(process_group->broadcast(flat_tensor_)) {}

  void finish() {
    work_->wait();

    // Copy the output of the broadcast operation back.
    auto output_tensors = torch::utils::unflatten_dense_tensors(
        flat_tensor_.front(), bucket_tensors_);
    TORCH_INTERNAL_ASSERT(output_tensors.size() == bucket_tensors_.size());
    for (size_t i = 0; i < output_tensors.size(); i++) {
      bucket_tensors_[i].copy_(output_tensors[i], /*non_blocking=*/true);
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

  // The broadcast work that is kicked off upon construction.
  std::shared_ptr<c10d::ProcessGroup::Work> work_;
};

} // namespace

// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    std::shared_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size) {
  // Coalesce tensors into buckets taking into account the maximum buffer size.
  // This routine is multi-device aware, so the tensors can be split across
  // multiple devices and can contain a mix of CPU and CUDA tensors.
  const auto buckets =
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

    in_flight.emplace_back(process_group, c10::fmap(bucket, lookup));
  }

  while (!in_flight.empty()) {
    in_flight.front().finish();
    in_flight.pop_front();
  }
}

} // namespace c10d

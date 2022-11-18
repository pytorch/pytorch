#include "ATen/cuda/CUDAEvent.h"
#ifdef USE_C10D_NCCL

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/SparseNCCLCollectives.hpp>

namespace c10d {

void check_gpu_tensors_different_devices_sparse(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(at::cuda::getNumGPUs())) {
    TORCH_CHECK(
        false,
        "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.is_cuda()) {
      TORCH_CHECK(false, "Tensors must be CUDA");
    }
    if (t.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      TORCH_CHECK(false, "Tensors must have identical size");
    }
    if (t.strides() != first.strides()) {
      TORCH_CHECK(false, "Tensors must have identical strides");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      TORCH_CHECK(false, "Tensors must be on distinct GPU devices");
    }
  }
}

class SparseAllReduceMetadata {
  public:
    SparseAllReduceMetadata(
      ProcessGroup* pg,
      int64_t sparseDims,
      int64_t denseDims,
      at::Tensor sparseTensor)
        : maxNnz_(0),
          allNnzs_(pg->getSize()),
          pg_(pg),
          sparseDims_(sparseDims),
          denseDims_(denseDims),
          sparseTensor_(sparseTensor) {

    }

    int64_t maxNnz() const {
      return maxNnz_;
    }

    const std::vector<int64_t> allNnzs() const {
      return allNnzs_;
    }

    ProcessGroup* pg() const {
      return pg_;
    }

    at::Tensor sparseTensor() const {
      return sparseTensor_;
    }

    int64_t sparseDims() const {
      return sparseDims_;
    }

    int64_t denseDims() const {
      return denseDims_;
    }

    void setMaxNnz(int64_t maxNnz) {
      maxNnz_ = maxNnz;
    }

    void setAllNnzs(const std::vector<int64_t>& allNnzs) {
      allNnzs_ = allNnzs;
    }

  private:
    // Maximum number of nnzs across all ranks.
    int64_t maxNnz_;
    // All nnzs across ranks.
    std::vector<int64_t> allNnzs_;
    // ProcessGroup to use.
    ProcessGroup *pg_;
    // Number of sparse_dims.
    int64_t sparseDims_;
    // Number of dense_dims.
    int64_t denseDims_;
    // SparseTensor to allreduce.
    at::Tensor sparseTensor_;
};

void validate_metadata(
  SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& all_metadata) {

  auto pg = metadata.pg();
  int64_t maxNnz;
  std::vector<int64_t> allNnzs(pg->getSize());

  // Validate metadata and compute nnzs.
  auto metadataToCompare = all_metadata[0][0].index({at::indexing::Slice(0, 9)});
  for (const auto i : c10::irange(pg->getSize())) {
    TORCH_CHECK(
      all_metadata[0][i].index({at::indexing::Slice(0, 9)}).equal(metadataToCompare),
      "Sparse Tensor Metadata is not consistent across ranks!")

    // TODO: Can we avoid this d2h sync?
    auto nnz = all_metadata[0][i][9].item<int64_t>();
    maxNnz = std::max(maxNnz, nnz);
    allNnzs[i] = nnz;
  }

  // Record the nnz information in our metadata.
  metadata.setMaxNnz(maxNnz);
  metadata.setAllNnzs(allNnzs);
}

c10::intrusive_ptr<Work> allgather_metadata(
  SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& gathered_metadata) {
    // Build tensor metadata, structure is as follows:
    // [0] -> dtype
    // [1:5] -> sparse_dims
    // [5:9] -> dense_dims
    // [9] -> nnz
    auto sparseTensor = metadata.sparseTensor();
    auto pg = metadata.pg();

    std::vector<int64_t> sparseMetadata(10, -1);
    sparseMetadata[0] = static_cast<int64_t>(sparseTensor.scalar_type());
    for (const auto i : c10::irange(4)) {
      if (i < metadata.sparseDims()) {
        sparseMetadata[i + 1] = sparseTensor.size(i);
      }

      if (i < metadata.denseDims()) {
        sparseMetadata[i + 5] = sparseTensor.size(i + metadata.sparseDims());
      }
    }
    sparseMetadata[9] = sparseTensor._nnz();

    auto options =
      c10::TensorOptions()
        .dtype(c10::kLong);

    at::Tensor sparseMetadataTensor =
      at::from_blob(sparseMetadata.data(), {10}, options).to(sparseTensor.device());

    // Allgather metadata for all sparse tensors and validate.
    std::vector<at::Tensor> output_placeholder;
    for (const auto i : c10::irange(pg->getSize())) {
      output_placeholder.emplace_back(at::empty_like(sparseMetadataTensor));
    }
    gathered_metadata.push_back(output_placeholder);

    std::vector<at::Tensor> input_metadata{sparseMetadataTensor};
    return pg->allgather(gathered_metadata, input_metadata);
}

void reshape_indices(
  const SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& outputIndices) {
  // Reshape outputIndices correctly by removing padding.
  auto allNnzs = metadata.allNnzs();
  for (const auto i : c10::irange(metadata.pg()->getSize())) {
    outputIndices[0][i] = outputIndices[0][i].narrow(1, 0, allNnzs[i]);
  }
}

c10::intrusive_ptr<Work> allgather_indices(
  const SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& outputIndices) {
    auto pg = metadata.pg();
    auto sparseTensor = metadata.sparseTensor();

    auto original_indices = sparseTensor.coalesce().indices();

    auto options =
      c10::TensorOptions()
        .dtype(original_indices.dtype())
        .device(original_indices.device());

    for (const auto i : c10::irange(pg->getSize())) {
      outputIndices[0][i] = at::empty({metadata.sparseDims(), metadata.maxNnz()}, options);
    }

    // Add padding.
    auto indices_padding = metadata.maxNnz() - original_indices.size(1);
    auto padded_indices = at::constant_pad_nd(original_indices, {0, indices_padding});
    std::vector<at::Tensor> input_indices{padded_indices};

    return pg->allgather(outputIndices, input_indices);
}

void reshape_values(
  const SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& outputValues) {
  // Reshape outputValues correctly by removing padding.
  auto allNnzs = metadata.allNnzs();
  for (const auto i : c10::irange(metadata.pg()->getSize())) {
    outputValues[0][i] = outputValues[0][i].narrow(0, 0, allNnzs[i]);
  }
}

c10::intrusive_ptr<Work> allgather_values(
  const SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& outputValues) {
    auto sparseTensor = metadata.sparseTensor();
    auto pg = metadata.pg();

    auto original_values = sparseTensor.coalesce().values();
    auto options =
      c10::TensorOptions()
        .dtype(original_values.dtype())
        .device(original_values.device());

    // Compute max size of values.
    std::vector<int64_t> max_values_size(metadata.denseDims()+ 1);
    max_values_size[0] = metadata.maxNnz();
    for (const auto i : c10::irange(metadata.denseDims())) {
      max_values_size[i + 1] = original_values.size(i + 1);
    }

    for (const auto i : c10::irange(pg->getSize())) {
      outputValues[0][i] = at::empty(max_values_size, options);
    }

    // Add padding.
    std::vector<int64_t> values_padding(2 * (metadata.denseDims() + 1), 0);
    values_padding[values_padding.size() - 1] = metadata.maxNnz() - original_values.size(0);
    auto padded_values = at::constant_pad_nd(original_values, values_padding);
    std::vector<at::Tensor> input_values{padded_values};
    return pg->allgather(outputValues, input_values);
}

void allgather_values_cb(
  const SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& outputIndices,
  std::vector<std::vector<at::Tensor>>& outputValues,
  at::cuda::CUDAEvent& endEvent,
  std::vector<at::cuda::CUDAStream>& workEndStreams) {
  reshape_values(metadata, outputValues);

  // Add all sparse tensors together.
  auto sparseTensor = metadata.sparseTensor();
  for (const auto i : c10::irange(metadata.pg()->getSize())) {
    // Accumulate tensors except self.
    if (i != metadata.pg()->getRank()) {
      sparseTensor += at::sparse_coo_tensor(
          outputIndices[0][i],
          outputValues[0][i],
          sparseTensor.sizes(),
          sparseTensor.options());
    }
  }

  // Coalesce before returning.
  sparseTensor.copy_(sparseTensor.coalesce());

  // Record appropriate CUDA events and streams for completion of work.
  auto currentStream = at::cuda::getCurrentCUDAStream(sparseTensor.device().index());
  endEvent.record(currentStream);
  workEndStreams.push_back(currentStream);
}

void allgather_indices_cb(
  const SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>>& outputIndices,
  at::cuda::CUDAEvent& endEvent,
  std::vector<at::cuda::CUDAStream>& workEndStreams) {
  reshape_indices(metadata, outputIndices);

  // Allgather values and reshape.
  std::vector<std::vector<at::Tensor>> outputValues{
      std::vector<at::Tensor>(metadata.pg()->getSize())};
  allgather_values(metadata, outputValues)
      ->getFuture()
      ->addCallback([&metadata, &outputValues, &outputIndices, &endEvent, &workEndStreams](
                        c10::ivalue::Future& future) {
        allgather_values_cb(metadata, outputIndices, outputValues, endEvent, workEndStreams);
      });
}

void allgather_metadata_cb(
  SparseAllReduceMetadata& metadata,
  std::vector<std::vector<at::Tensor>> gathered_metadata,
  at::cuda::CUDAEvent& endEvent,
  std::vector<at::cuda::CUDAStream>& workEndStreams) {
  validate_metadata(metadata, gathered_metadata);

  // Allgather indices and reshape.
  std::vector<std::vector<at::Tensor>> outputIndices{
      std::vector<at::Tensor>(metadata.pg()->getSize())};
  allgather_indices(metadata, outputIndices)
      ->getFuture()
      ->addCallback([&metadata, &outputIndices, &endEvent, &workEndStreams](c10::ivalue::Future& future) {
        allgather_indices_cb(metadata, outputIndices, endEvent, workEndStreams);
      });
}

c10::intrusive_ptr<Work> sparse_allreduce(
    ProcessGroup* pg,
    std::vector<at::Tensor>& sparseTensors,
    const AllreduceOptions& opts,
    bool desyncDebug) {
  // Perform validation.
  check_gpu_tensors_different_devices_sparse(sparseTensors);

  auto world_size = pg->getSize();
  auto rank = pg->getRank();

  // Build work object to be returned.
  std::vector<c10::Device> devices;
  for (const auto& sparseTensor : sparseTensors) {
    devices.push_back(sparseTensor.device());
  }
  auto work = c10::make_intrusive<ProcessGroupNCCL::WorkNCCL>(
      devices,
      rank,
      OpType::ALLREDUCE,
      pg->getSequenceNumberForGroup(),
      "nccl:sparse_all_reduce",
      sparseTensors,
      desyncDebug);

  // End events for entire work.
  auto endEvents = work->endEvents();

  // Streams on which work eventually completed.
  std::vector<at::cuda::CUDAStream> workEndStreams;

  for (const auto idx : c10::irange(sparseTensors.size())) {
    auto sparseTensor = sparseTensors[idx];

    // Basic validation.
    const auto sparseDims = sparseTensor.sparse_dim();
    const auto denseDims = sparseTensor.dense_dim();
    TORCH_CHECK(
      sparseDims <= 4 && denseDims <= 4,
      "SparseTensor allreduce only supported for sparse and dense dims <= 4"
    )


    // Allgather metadata and validate.
    SparseAllReduceMetadata metadata(pg, sparseDims, denseDims, sparseTensor);
    std::vector<std::vector<at::Tensor>> gathered_metadata;
    // This begins a chain of callbacks to ensure complete asynchronous
    // execution and to avoid blocking the current CUDA stream on this work.
    auto& endEvent = (*endEvents)[idx];
    allgather_metadata(metadata, gathered_metadata)->getFuture()->addCallback(
      [&metadata, &gathered_metadata, &endEvent, &workEndStreams](c10::ivalue::Future& future) {
        allgather_metadata_cb(metadata, gathered_metadata, endEvent, workEndStreams);
      }
    );

  }

  // Mark future completed appropriately.
  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(workEndStreams);
    work->createFuture();
    work->getFuture()->markCompleted(at::IValue(sparseTensors));
  }

  return work;
}

} // namespace c10d

#endif // USE_C10D_NCCL

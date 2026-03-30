#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>

namespace c10d {
namespace intra_node_comm {

static constexpr size_t kOneShotThreshBytes = 256 * 1024;
static constexpr size_t kTwoShotThreshBytes = 10 * 1024 * 1024;

static void checkInput(const at::Tensor& input, int deviceIdx) {
  TORCH_CHECK(
      input.dtype() == at::kBFloat16 || input.dtype() == at::kFloat,
      "oneShotAllReduce only supports float and bf16 for now");
  TORCH_CHECK(input.is_non_overlapping_and_dense());
  TORCH_CHECK(input.device().is_cuda());
  TORCH_CHECK(
      input.get_device() == deviceIdx,
      "IntraNodeComm: expect input to be on device ",
      deviceIdx,
      ", got device ",
      input.get_device());
}

bool isIntraNodeCommSupported() {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  return false;
#else
  return true;
#endif
}

at::Tensor IntraNodeComm::oneShotAllReduce(
    const at::Tensor& input,
    at::cuda::CUDAStream& stream) {
  checkInput(input, deviceIdx_);

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::one_shot_all_reduce_out", "")
                .typed<at::Tensor(
                    const at::Tensor&, std::string, std::string, at::Tensor)>();

  auto symmMemTensor = at::from_blob(
      symmetricMemoryPtr_,
      input.sizes(),
      at::TensorOptions().dtype(input.dtype()).device(input.device()));

  symmMemTensor.copy_(input);
  op.call(symmMemTensor, "sum", "", input);
  return input;
}

at::Tensor IntraNodeComm::twoShotAllReduce(
    const at::Tensor& input,
    at::cuda::CUDAStream& stream) {
  checkInput(input, deviceIdx_);

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::two_shot_all_reduce_", "")
                .typed<at::Tensor(at::Tensor, std::string, std::string)>();

  auto symmMemTensor = at::from_blob(
      symmetricMemoryPtr_,
      input.sizes(),
      at::TensorOptions().dtype(input.dtype()).device(input.device()));

  symmMemTensor.copy_(input);
  op.call(symmMemTensor, "sum", "");
  input.copy_(symmMemTensor);
  return input;
}

AllReduceAlgo IntraNodeComm::selectAllReduceAlgo(const at::Tensor& input) {
  // Only support float and bf16 for now
  if (input.dtype() != at::kBFloat16 && input.dtype() != at::kFloat) {
    return AllReduceAlgo::NONE;
  }
  const auto inputSize =
      static_cast<size_t>(input.numel() * input.element_size());
  const size_t ptrAlignment = get_alignment(
      static_cast<size_t>(input.storage_offset() * input.element_size()));
  const size_t sizeAlignment = get_alignment(inputSize);
  const size_t alignment = std::min(ptrAlignment, sizeAlignment);

  if (topology_ == Topology::FULLY_CONNECTED) {
    // Both symm_mem::one_shot_all_reduce and symm_mem::two_shot_all_reduce_
    // currently requires the input to be at least 4-bytes aligned.
    if (alignment >= 4 && inputSize <= kOneShotThreshBytes &&
        inputSize <= bufferSize_) {
      return AllReduceAlgo::ONE_SHOT;
    }
    if (alignment >= 4 && inputSize <= kTwoShotThreshBytes &&
        inputSize <= bufferSize_) {
      return AllReduceAlgo::TWO_SHOT;
    }
  }
  return AllReduceAlgo::NONE;
}

static int64_t usageCounter = 0;

at::Tensor IntraNodeComm::allReduce(
    const at::Tensor& input,
    AllReduceAlgo algo) {
  // Report usage for testing purposes.
  // We don't care about overflowing.
  ++usageCounter;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (algo) {
    case AllReduceAlgo::ONE_SHOT:
      return oneShotAllReduce(input, stream);
    case AllReduceAlgo::TWO_SHOT:
      return twoShotAllReduce(input, stream);
    default:
      C10_THROW_ERROR(ValueError, "IntraNodeComm: invalid algo");
  }
}

int64_t getIntraNodeCommUsageCounter() {
  return usageCounter;
}

} // namespace intra_node_comm
} // namespace c10d

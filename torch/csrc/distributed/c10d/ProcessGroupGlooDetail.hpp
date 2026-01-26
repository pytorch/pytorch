#pragma once

#ifdef USE_C10D_GLOO

#include <c10/util/Registry.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include <gloo/allgather.h>
#include <gloo/allgatherv.h>
#include <gloo/allreduce.h>
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/scatter.h>

#ifdef _WIN32
#define GENERATE_ALL_TYPES(type, func, ...)      \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(__VA_ARGS__);                  \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<c10::Half>(__VA_ARGS__);              \
      break;                                     \
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(__VA_ARGS__);          \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(__VA_ARGS__);                \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }

#define HOST_NAME_MAX 256
#else
#define GENERATE_ALL_TYPES(type, func, args...)  \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(args);                         \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(args);                        \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<c10::Half>(args);                     \
      break;                                     \
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(args);                 \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(args);                        \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(args);                       \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }
#endif

namespace c10d {

TORCH_DECLARE_TYPED_REGISTRY(
    GlooAllreduceRegistry,
    c10::DeviceType,
    ProcessGroupGloo::AsyncWork,
    c10::intrusive_ptr,
    std::shared_ptr<gloo::Context>,
    std::vector<at::Tensor>&,
    ReduceOp,
    uint32_t,
    uint64_t,
    std::chrono::milliseconds);

// This function initializes a vector of CUDA streams, one for every
// tensor in the input tensor vector, and ensures that these streams are
// synchronized with the current default streams. This is needed so
// that new work on the new streams is serialized w.r.t. all operations
// on the tensors.
TORCH_API void initializeStreamsEvents(
    const std::vector<at::Tensor>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events);

// This function initializes a vector of CUDA streams, one per device,
// and ensures that these streams are synchronized with the current default
// streams. It is assumed that the tensors in the nested tensor vectors are
// on the same device.
TORCH_API void initializeStreamsEvents(
    std::vector<std::vector<at::Tensor>>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events);

typedef void (*ReduceFunc)(void*, const void*, const void*, size_t);

template <typename T, std::enable_if_t<!std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
    case ReduceOp::AVG:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::BAND:
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with non-integral dtype");
      break;
    case ReduceOp::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with non-integral dtype");
      break;
    case ReduceOp::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with non-integral dtype");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
    default:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

// Bitwise AND with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void band(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] & tb[i];
  }
}

// Bitwise OR with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] | tb[i];
  }
}

// Bitwise XOR with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bxor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] ^ tb[i];
  }
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
    case ReduceOp::AVG:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::BAND:
      return ReduceFunc(&band<T>);
    case ReduceOp::BOR:
      return ReduceFunc(&bor<T>);
    case ReduceOp::BXOR:
      return ReduceFunc(&bxor<T>);
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
    default:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

template <typename T, typename O>
void setInputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutputs(O& opts, std::vector<at::Tensor>& tensors, int64_t count) {
  opts.setOutputs(getDataPointers<T>(tensors), count);
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

static at::Tensor pinnedLike(at::Tensor& tensor) {
  auto* allocator = at::detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      static_cast<int64_t>(at::detail::computeStorageNbytes(
          tensor.sizes(), tensor.strides(), tensor.dtype().itemsize())),
      allocator,
      /*resizable=*/false);
  return at::empty({0}, tensor.options().device(at::kCPU))
      .set_(storage, 0, tensor.sizes(), tensor.strides());
}

class AsyncAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            {inputs},
            OpType::ALLREDUCE,
            seq,
            timeout,
            "gloo:all_reduce",
            inputs),
        inputs(inputs),
        reduceOp(std::move(reduceOp)),
        tag(tag) {}

  std::vector<at::Tensor> inputs;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduce(std::vector<at::Tensor>& tensors) {
    auto tensor = tensors[0];
    if (tensor.is_complex()) {
      TORCH_CHECK(
          c10d::isComplexViewAsRealAllowed(reduceOp),
          "all_reduce does not support",
          reduceOp,
          "on complex tensors");
      tensor = at::view_as_real(tensor);
    }
    gloo::AllreduceOptions opts(context_);
    const auto& scalarType = tensor.scalar_type();
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    // Use tensor.numel() instead of tensors[0].numel() to
    // get the right number of elements when tensors[0] is complex
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors, tensor.numel());
    gloo::allreduce(opts);

    // Gloo doesn't support AVG so we use SUM + division.
    if (reduceOp == ReduceOp::AVG) {
      tensors[0] /= context_->size;
    }
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return inputs;
  }

  void run() override {
    allreduce(inputs);
  }

  template <typename T>
  void getFunction(gloo::AllreduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::AllreduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp& op) {
    gloo::AllreduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

class AsyncAllreduceCoalescedWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCoalescedWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncAllreduceWork(
            context,
            inputs,
            std::move(reduceOp),
            tag,
            seq,
            timeout) {}

  void run() override {
    allreduceCoalesced(inputs);
  }

 private:
  void allreduceCoalesced(std::vector<at::Tensor>& tensors) {
    // reduce coalesced, flattened tensors.
    at::Tensor coalescedTensor = flattenDenseTensors(tensors);
    std::vector<at::Tensor> allreduceInput = {coalescedTensor};
    allreduce(allreduceInput);

    // separate and reshape tensors.
    size_t offset = 0;
    for (at::Tensor& tensor : tensors) {
      const int64_t tensorNumel = tensor.numel();
      const c10::IntArrayRef tensorShape = tensor.sizes();
      tensor.copy_(coalescedTensor.slice(0, offset, offset + tensorNumel)
                       .view(tensorShape));
      offset += tensorNumel;
    }
  }
};

class AsyncSparseAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncSparseAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            {inputs},
            OpType::_ALLREDUCE_SPARSE,
            seq,
            timeout,
            "gloo:sparse_all_reduce",
            inputs),
        inputs(inputs),
        tag(tag) {}

  std::vector<at::Tensor> inputs;
  const uint32_t tag;

  // We share dimensionality about the sparse tensors before collecting
  // their contents. We assume here that the maximum number of sparse
  // and dense dimensions is 4. This is stored in a contiguous piece of
  // memory so that we can easily run allgather on it.
  //
  // The layout of this memory is as follows:
  //
  //   - [0:4]: sparse dims
  //   - [4:8]: dense dims
  //   -   [8]: nnz
  //
  class SparseTensorMetadata {
   public:
    static constexpr auto dim = 9;

    // Construct from an existing metadata tensor to facilitate structured
    // access to metadata from peers, after gathering it.
    explicit SparseTensorMetadata(at::Tensor metadata)
        : metadata_(std::move(metadata)),
          data_(metadata_.mutable_data_ptr<int64_t>()) {
      AT_ASSERT(metadata_.scalar_type() == at::kLong);
      AT_ASSERT(metadata_.dim() == 1);
      AT_ASSERT(metadata_.size(0) == dim);
    }

    // Populate the metadata.
    void populate_from_sparse_tensor(const at::Tensor& tensor) {
      const auto sparse_dim = tensor.sparse_dim();
      AT_ASSERT(sparse_dim <= 4);
      for (const auto i : c10::irange(4)) {
        if (i < sparse_dim) {
          data_[i] = tensor.size(i);
        }
      }
      const auto dense_dim = tensor.dense_dim();
      AT_ASSERT(dense_dim <= 4);
      for (const auto i : c10::irange(4)) {
        if (i < dense_dim) {
          data_[i + 4] = tensor.size(sparse_dim + i);
        }
      }
      data_[8] = tensor._nnz();
    }

    std::vector<int64_t> sizes() const {
      std::vector<int64_t> sizes;
      // Sparse sizes
      for (const auto i : c10::irange(4)) {
        if (data_[i] <= 0) {
          break;
        }
        sizes.push_back(data_[i]);
      }
      // Dense sizes
      for (const auto i : c10::irange(4, 8)) {
        if (data_[i] <= 0) {
          break;
        }
        sizes.push_back(data_[i]);
      }
      return sizes;
    }

    int64_t nnz() const {
      return data_[8];
    }

   protected:
    at::Tensor metadata_;
    int64_t* data_;
  };

  // Sparse allreduce is implemented with allgather on indices and values.
  // Every process then sums the resulting sparse tensors locally.
  // The nnz for sparse tensors may be different across processes, so first
  // we run allgather on the nnz, and then allgather with max(nnz).
  at::Tensor allreduce(std::vector<at::Tensor>& tensors) {
    // TODO: This is a massive hack!  There is some confusion about
    // Variable/Tensor inside the body of this function.  Turning off
    // grad smooths over the confusion for now.  This fixes
    // test/test_c10d_gloo.py ProcessGroupGlooTest.test_sparse_allreduce_basics
    //
    // The correct fix is to stop allocating tensors that are not variables,
    // but to conveniently do this c10d must depend on torch not ATen
    at::AutoDispatchBelowAutograd guard;
    auto input = tensors[0];

    // Perform local reduction if we have multiple inputs.
    for (const auto i : c10::irange(1, tensors.size())) {
      input += tensors[i];
    }

    // Need to coalesce before we can access indices and values.
    input = input.coalesce();

    // Gather metadata information from all ranks.
    auto metadata = allgather_metadata(input);

    // Sanity check dimensionality across ranks.
    {
      const auto expected = metadata[context_->rank].sizes();
      for (const auto i : c10::irange(context_->size)) {
        if (i == context_->rank) {
          continue;
        }
        const auto actual = metadata[i].sizes();
        TORCH_CHECK(actual == expected, "Sparse dimensions do not match");
      }
    }

    // Gather all indices and all values.
    auto indices = allgather_indices(input, metadata);
    auto values = allgather_values(input, metadata);

    // Perform global reduction.
    AT_ASSERT(static_cast<int>(indices.size()) == context_->size);
    AT_ASSERT(static_cast<int>(values.size()) == context_->size);
    auto output = at::sparse_coo_tensor(
        indices[0], values[0], input.sizes(), input.options());
    for (const auto i : c10::irange(1, context_->size)) {
      output += at::sparse_coo_tensor(
          indices[i], values[i], input.sizes(), input.options());
    }

    // Coalesce for good measure.
    return output.coalesce();
  }

  void run() override {
    auto output = allreduce(inputs);

    // This copy is needed when we run a multi-gpu version of reduce (multiple
    // inputs per rank).
    for (const auto i : c10::irange(inputs.size())) {
      inputs[i].copy_(output);
    }
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return inputs;
  }

 private:
  std::vector<SparseTensorMetadata> allgather_metadata(
      const at::Tensor& tensor) {
    auto buffer =
        at::zeros({context_->size, SparseTensorMetadata::dim}, at::kLong);

    // Prepare metadata vector (1 entry per rank)
    std::vector<SparseTensorMetadata> metadata;
    metadata.reserve(context_->size);
    for (const auto i : c10::irange(context_->size)) {
      metadata.emplace_back(buffer.select(0, i));
    }

    // Populate data for this rank
    metadata[context_->rank].populate_from_sparse_tensor(tensor);

    // Allgather metadata
    gloo::AllgatherOptions opts(context_);
    opts.setOutput(buffer.mutable_data_ptr<int64_t>(), buffer.numel());
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    gloo::allgather(opts);

    return metadata;
  }

  std::vector<at::Tensor> allgather_indices(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata) {
    const auto sparseDim = tensor.sparse_dim();

    std::vector<size_t> counts(context_->size);
    size_t totalSize = 0;
    for (const auto i : c10::irange(metadata.size())) {
      counts[i] = metadata[i].nnz() * sparseDim;
      totalSize += counts[i];
    }

    auto output = at::empty({static_cast<int64_t>(totalSize)}, at::kLong);

    // tensors copied from cuda may not be contiguous, get a contiguous
    // tensor before use its data_ptr
    auto input = tensor.indices().contiguous();

    // Allgatherv indices.
    gloo::AllgathervOptions opts(context_);
    opts.setInput(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<int64_t*>(input.const_data_ptr<int64_t>()),
        input.numel());
    opts.setOutput(output.mutable_data_ptr<int64_t>(), counts);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    gloo::allgatherv(opts);

    // Compile indices tensor per rank.
    std::vector<at::Tensor> indices;
    indices.reserve(metadata.size());
    int64_t offset = 0;
    for (const auto& i : metadata) {
      const auto nnz = i.nnz();
      const auto numel = sparseDim * nnz;
      indices.push_back(
          output.narrow(0, offset, numel).reshape({sparseDim, nnz}));
      offset += numel;
    }

    return indices;
  }

  std::vector<at::Tensor> allgather_values(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata) {
    // There are nnz #dense_dim()-dimensional tensors per rank.
    const auto valueShape = tensor.sizes().slice(tensor.sparse_dim());
    int64_t denseNumel = 1;
    for (auto dim : valueShape) {
      denseNumel *= dim;
    }

    std::vector<size_t> counts(context_->size);
    int64_t totalSize = 0;
    for (const auto i : c10::irange(metadata.size())) {
      counts[i] = metadata[i].nnz() * denseNumel;
      totalSize += static_cast<int64_t>(counts[i]);
    }

    auto output = at::empty({totalSize}, tensor.scalar_type());

    // Allgatherv indices.
    gloo::AllgathervOptions opts(context_);
    // tensors copied from cuda may not be contiguous, get a contiguous
    // tensor before use its data_ptr
    at::Tensor valueTensor = tensor.values().contiguous();
    GENERATE_ALL_TYPES(valueTensor.scalar_type(), setInput, opts, valueTensor);
    GENERATE_ALL_TYPES(
        valueTensor.scalar_type(), setOutput, opts, output, counts);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    gloo::allgatherv(opts);

    // Compile values tensor per rank.
    std::vector<at::Tensor> values;
    values.reserve(metadata.size());
    int64_t offset = 0;
    for (const auto& i : metadata) {
      const auto nnz = i.nnz();
      const auto numel = denseNumel * nnz;
      auto tensorShape = std::vector<int64_t>({(int64_t)nnz});
      std::copy(
          valueShape.begin(),
          valueShape.end(),
          std::back_inserter(tensorShape));
      values.push_back(output.narrow(0, offset, numel).reshape(tensorShape));
      offset += numel;
    }

    return values;
  }
};

} // namespace c10d

#endif

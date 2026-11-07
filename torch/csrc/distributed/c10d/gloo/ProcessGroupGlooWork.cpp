#include <torch/csrc/distributed/c10d/gloo/ProcessGroupGlooDetail.hpp>

#ifdef USE_C10D_GLOO

#include <torch/csrc/distributed/c10d/Utils.hpp>

namespace c10d {
namespace {

template <typename T>
void getAllreduceFunction(
    gloo::AllreduceOptions::Func& fn,
    const ReduceOp op) {
  fn = toFunction<T>(op);
}

gloo::AllreduceOptions::Func getAllreduceFunction(
    const at::ScalarType& dtype,
    const ReduceOp& op) {
  gloo::AllreduceOptions::Func fn;
  GENERATE_ALL_TYPES(dtype, getAllreduceFunction, fn, op);
  return fn;
}

template <typename T>
void getReduceFunction(gloo::ReduceOptions::Func& fn, const ReduceOp op) {
  fn = toFunction<T>(op);
}

gloo::ReduceOptions::Func getReduceFunction(
    const at::ScalarType& dtype,
    const ReduceOp& op) {
  gloo::ReduceOptions::Func fn;
  GENERATE_ALL_TYPES(dtype, getReduceFunction, fn, op);
  return fn;
}

} // namespace

AsyncBroadcastWork::AsyncBroadcastWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    int rootRank,
    int rootTensor,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {inputs}, OpType::BROADCAST, seq, timeout,
          "gloo:broadcast", inputs),
      inputs(inputs),
      rootRank(rootRank),
      rootTensor(rootTensor),
      tag(tag) {}

const std::vector<at::Tensor> AsyncBroadcastWork::getInputTensors() {
  return inputs;
}

const std::vector<at::Tensor> AsyncBroadcastWork::getOutputTensors() {
  return inputs;
}

void AsyncBroadcastWork::broadcast(at::Tensor tensor) {
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  const auto& scalarType = tensor.scalar_type();
  gloo::BroadcastOptions opts(context_);
  opts.setRoot(rootRank);
  opts.setTag(tag);
  opts.setTimeout(getTimeout());
  GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
  gloo::broadcast(opts);
}

void AsyncBroadcastWork::run() {
  broadcast(inputs[rootTensor]);
  for (const auto i : c10::irange(inputs.size())) {
    if (i != static_cast<size_t>(rootTensor)) {
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
}

AsyncAllreduceWork::AsyncAllreduceWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : AsyncAllreduceWork(
          std::move(context), inputs, reduceOp, tag, seq, OpType::ALLREDUCE,
          timeout) {}

AsyncAllreduceWork::AsyncAllreduceWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    OpType opType,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {inputs}, opType, seq, timeout,
          "gloo:all_reduce", inputs),
      inputs(inputs),
      reduceOp(reduceOp),
      tag(tag) {}

const std::vector<at::Tensor> AsyncAllreduceWork::getInputTensors() {
  return inputs;
}

const std::vector<at::Tensor> AsyncAllreduceWork::getOutputTensors() {
  return inputs;
}

void AsyncAllreduceWork::allreduce(std::vector<at::Tensor>& tensors) {
  auto tensor = tensors[0];
  if (tensor.is_complex()) {
    TORCH_CHECK(
        c10d::isComplexViewAsRealAllowed(reduceOp),
        "all_reduce does not support", reduceOp, "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  gloo::AllreduceOptions opts(context_);
  const auto& scalarType = tensor.scalar_type();
opts.setReduceFunction(getAllreduceFunction(scalarType, reduceOp));
  opts.setTimeout(getTimeout());
  GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors, tensor.numel());
  gloo::allreduce(opts);
  if (reduceOp == ReduceOp::AVG) {
    tensors[0] /= context_->size;
  }
}

void AsyncAllreduceWork::run() {
  allreduce(inputs);
}

AsyncSparseAllreduceWork::AsyncSparseAllreduceWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {inputs}, OpType::_ALLREDUCE_SPARSE, seq, timeout,
          "gloo:sparse_all_reduce", inputs),
      inputs(inputs),
      tag(tag) {}

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
class AsyncSparseAllreduceWork::SparseTensorMetadata {
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
at::Tensor AsyncSparseAllreduceWork::allreduce(std::vector<at::Tensor>& tensors) {
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
  auto metadata = allgatherMetadata(input);

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
  auto indices = allgatherIndices(input, metadata);
  auto values = allgatherValues(input, metadata);

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

void AsyncSparseAllreduceWork::run() {
  auto output = allreduce(inputs);

  // This copy is needed when we run a multi-gpu version of reduce (multiple
  // inputs per rank).
  for (const auto& input : inputs) {
    input.copy_(output);
  }
}

const std::vector<at::Tensor> AsyncSparseAllreduceWork::getInputTensors() {
  return inputs;
}

const std::vector<at::Tensor> AsyncSparseAllreduceWork::getOutputTensors() {
  return inputs;
}

std::vector<AsyncSparseAllreduceWork::SparseTensorMetadata> AsyncSparseAllreduceWork::allgatherMetadata(
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

std::vector<at::Tensor> AsyncSparseAllreduceWork::allgatherIndices(
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
  // tensor before using its data_ptr
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

std::vector<at::Tensor> AsyncSparseAllreduceWork::allgatherValues(
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

  // Allgatherv values.
  gloo::AllgathervOptions opts(context_);
  // tensors copied from cuda may not be contiguous, get a contiguous
  // tensor before using its data_ptr
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

AsyncAllreduceCoalescedWork::AsyncAllreduceCoalescedWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : AsyncAllreduceWork(
          std::move(context), inputs, reduceOp, tag, seq, timeout) {}

void AsyncAllreduceCoalescedWork::allreduceCoalesced(
    std::vector<at::Tensor>& tensors) {
  at::Tensor coalescedTensor = flattenDenseTensors(tensors);
  std::vector<at::Tensor> allreduceInput = {coalescedTensor};
  allreduce(allreduceInput);
  size_t offset = 0;
  for (at::Tensor& tensor : tensors) {
    const int64_t tensorNumel = tensor.numel();
    const c10::IntArrayRef tensorShape = tensor.sizes();
    tensor.copy_(coalescedTensor.slice(0, offset, offset + tensorNumel)
                     .view(tensorShape));
    offset += tensorNumel;
  }
}

void AsyncAllreduceCoalescedWork::run() {
  allreduceCoalesced(inputs);
}

AsyncReduceWork::AsyncReduceWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    int rootRank,
    int rootTensor,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {inputs}, OpType::REDUCE, seq, timeout,
          "gloo:reduce", inputs),
      inputs(inputs),
      rootRank(rootRank),
      rootTensor(rootTensor),
      reduceOp(reduceOp),
      tag(tag) {}

const std::vector<at::Tensor> AsyncReduceWork::getInputTensors() {
  return inputs;
}

const std::vector<at::Tensor> AsyncReduceWork::getOutputTensors() {
  return inputs;
}

void AsyncReduceWork::reduce(std::vector<at::Tensor>& tensors) {
  auto tensor = tensors[0];
  if (tensor.is_complex()) {
    TORCH_CHECK(
        c10d::isComplexViewAsRealAllowed(reduceOp),
        "reduce does not support", reduceOp, "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  gloo::ReduceOptions opts(context_);
  const auto& scalarType = tensor.scalar_type();
  opts.setRoot(rootRank);
  opts.setTag(tag);
opts.setReduceFunction(getReduceFunction(scalarType, reduceOp));
  GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
  gloo::reduce(opts);
  if (reduceOp == ReduceOp::AVG) {
    tensors[0] /= context_->size;
  }
}

void AsyncReduceWork::run() {
  reduce(inputs);
}

AsyncAllgatherWork::AsyncAllgatherWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), outputs, OpType::ALLGATHER, seq, timeout,
          "gloo:all_gather", inputs),
      outputs(outputs),
      inputs(inputs),
      tag(tag) {}

const std::vector<at::Tensor> AsyncAllgatherWork::getInputTensors() {
  return inputs;
}

const std::vector<at::Tensor> AsyncAllgatherWork::getOutputTensors() {
  return {newLikeFlat(outputs[0])};
}

void AsyncAllgatherWork::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs) {
  const auto& scalarType = inputs[0].scalar_type();
  gloo::AllgatherOptions opts(context_);
  opts.setTag(tag);
  opts.setTimeout(getTimeout());
  at::Tensor flatInputTensor = flattenDenseTensors(inputs);
  GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);
  at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
  GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
  gloo::allgather(opts);
  for (auto& outputgroup : outputs) {
    for (const auto j : c10::irange(outputgroup.size())) {
      outputgroup[j].copy_(flatOutputTensor[static_cast<int64_t>(j)]);
    }
  }
}

void AsyncAllgatherWork::run() {
  allgather(outputs, inputs);
}

AsyncAllgatherCoalescedWork::AsyncAllgatherCoalescedWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), outputs, OpType::ALLGATHER_COALESCED, seq, timeout,
          "gloo:all_gather", inputs),
      outputs(outputs),
      inputs(inputs),
      tag(tag) {}

const std::vector<at::Tensor> AsyncAllgatherCoalescedWork::getInputTensors() {
  return inputs;
}

const std::vector<at::Tensor> AsyncAllgatherCoalescedWork::getOutputTensors() {
  return {newLikeFlat(outputs[0])};
}

void AsyncAllgatherCoalescedWork::allgatherCoalesced() {
  TORCH_INTERNAL_ASSERT(!outputs.empty() && !outputs[0].empty() && !inputs.empty());
  const auto& scalarType = inputs[0].scalar_type();
  gloo::AllgatherOptions opts(context_);
  opts.setTag(tag);
  opts.setTimeout(getTimeout());
  at::Tensor flatInputTensor = flattenDenseTensors(inputs);
  GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);
  int64_t output_numel = 0;
  for (const auto& tensor : outputs[0]) {
    output_numel += tensor.numel();
  }
  output_numel *= static_cast<int64_t>(outputs.size());
  at::Tensor flatOutputTensor = at::empty({output_numel}, outputs[0][0].options());
  GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
  gloo::allgather(opts);
  int64_t current_element = 0;
  for (auto& output_list : outputs) {
    for (auto& output_tensor : output_list) {
      output_tensor.copy_(flatOutputTensor.narrow(0, current_element, output_tensor.numel()).reshape(output_tensor.sizes()), true);
      current_element += output_tensor.numel();
    }
  }
}

void AsyncAllgatherCoalescedWork::run() {
  allgatherCoalesced();
}

AsyncGatherWork::AsyncGatherWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    int root,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), outputs, OpType::GATHER, seq, timeout,
          "gloo:gather", inputs),
      outputs(outputs), inputs(inputs), root(root), tag(tag) {}

const std::vector<at::Tensor> AsyncGatherWork::getInputTensors() { return inputs; }
const std::vector<at::Tensor> AsyncGatherWork::getOutputTensors() {
  return outputs.empty() ? std::vector<at::Tensor>{}
                         : std::vector<at::Tensor>{newLikeFlat(outputs[0])};
}

void AsyncGatherWork::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs) {
  const auto scalarType = inputs[0].scalar_type();
  gloo::GatherOptions opts(context_);
  opts.setRoot(root);
  opts.setTag(tag);
  opts.setTimeout(getTimeout());
  at::Tensor flatOutputTensor;
  if (context_->rank == root) {
    flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
  }
  at::Tensor flatInputTensor = flattenDenseTensors(inputs[0]);
  GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);
  gloo::gather(opts);
  if (context_->rank == root) {
    for (const auto i : c10::irange(outputs[0].size())) {
      outputs[0][i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);
    }
  }
}

void AsyncGatherWork::run() { gather(outputs, inputs); }

AsyncScatterWork::AsyncScatterWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    int root,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {outputs}, OpType::SCATTER, seq, timeout,
          "gloo:scatter", inputs.empty() ? std::nullopt
                                          : std::optional<std::vector<at::Tensor>>(inputs[0])),
      outputs(outputs), inputs(inputs), root(root), tag(tag) {}

const std::vector<at::Tensor> AsyncScatterWork::getInputTensors() {
  return inputs.empty() ? std::vector<at::Tensor>{}
                        : std::vector<at::Tensor>{newLikeFlat(inputs[0])};
}
const std::vector<at::Tensor> AsyncScatterWork::getOutputTensors() { return outputs; }

void AsyncScatterWork::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs) {
  const auto scalarType = outputs[0].scalar_type();
  gloo::ScatterOptions opts(context_);
  opts.setRoot(root);
  opts.setTag(tag);
  opts.setTimeout(getTimeout());
  if (context_->rank == root) {
    GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
  }
  GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
  gloo::scatter(opts);
}

void AsyncScatterWork::run() { scatter(outputs, inputs); }

AsyncAlltoallWork::AsyncAlltoallWork(
    std::shared_ptr<gloo::Context> context,
    at::Tensor& output,
    at::Tensor& input,
    std::vector<int64_t>& outputCounts,
    std::vector<int64_t>& inputCounts,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {{output}}, OpType::ALLTOALL, seq, timeout,
          "gloo:all_to_all", std::optional<std::vector<at::Tensor>>({input})),
      output(output), input(input), outputCounts(std::move(outputCounts)),
      inputCounts(std::move(inputCounts)), tag(tag) {}
const std::vector<at::Tensor> AsyncAlltoallWork::getInputTensors() { return {input}; }
const std::vector<at::Tensor> AsyncAlltoallWork::getOutputTensors() { return {output}; }

void AsyncAlltoallWork::alltoall(at::Tensor& outputTensor, at::Tensor& inputTensor) {
  const auto scalarType = outputTensor.scalar_type();
  if (outputCounts.empty() && inputCounts.empty()) {
    gloo::AlltoallOptions opts(context_);
    opts.setTag(tag); opts.setTimeout(getTimeout());
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);
    gloo::alltoall(opts);
  } else {
    std::vector<int64_t> sendCounts(context_->size), recvCounts(context_->size);
    std::vector<int64_t> sendOffsets(context_->size), recvOffsets(context_->size);
    c10d::computeLengthsAndOffsets(inputCounts, inputTensor, &sendCounts, &sendOffsets);
    c10d::computeLengthsAndOffsets(outputCounts, outputTensor, &recvCounts, &recvOffsets);
    gloo::AlltoallvOptions opts(context_);
    opts.setTag(tag); opts.setTimeout(getTimeout());
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor, sendCounts);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor, recvCounts);
    gloo::alltoallv(opts);
  }
}
void AsyncAlltoallWork::run() { alltoall(output, input); }

AsyncAlltoallListWork::AsyncAlltoallListWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout)
    : ProcessGroupGlooAsyncWork(
          std::move(context), {outputs}, OpType::ALLTOALL, seq, timeout,
          "gloo:all_to_all", inputs),
      outputs(outputs), inputs(inputs), tag(tag) {}
const std::vector<at::Tensor> AsyncAlltoallListWork::getInputTensors() { return inputs; }
const std::vector<at::Tensor> AsyncAlltoallListWork::getOutputTensors() { return outputs; }

void AsyncAlltoallListWork::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors) {
  const auto scalarType = inputTensors[0].scalar_type();
  gloo::AlltoallOptions opts(context_);
  opts.setTag(tag); opts.setTimeout(getTimeout());
  at::Tensor flatInputTensor = flattenDenseTensors(inputTensors);
  GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);
  at::Tensor flatOutputTensor = newLikeFlat(outputTensors);
  GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
  gloo::alltoall(opts);
  for (const auto i : c10::irange(outputTensors.size())) {
    outputTensors[i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);
  }
}
void AsyncAlltoallListWork::run() { alltoall(outputs, inputs); }

} // namespace c10d

#endif // USE_C10D_GLOO

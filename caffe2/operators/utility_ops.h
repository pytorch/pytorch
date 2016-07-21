#ifndef CAFFE2_OPERATORS_UTILITY_OPS_H_
#define CAFFE2_OPERATORS_UTILITY_OPS_H_

#include <fstream>
#include <sstream>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

const char kPrintFileExtension[] = ".log";

template <class Context>
class PrintOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;
  PrintOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        to_file_(OperatorBase::GetSingleArgument<int>("to_file", 0)),
        limit_(OperatorBase::GetSingleArgument<int>("limit", 0)) {
    if (limit_ == 0) {
      limit_ = INT_MAX;
    }
    if (to_file_) {
      // We will output to file instead of printing on screen.
      const string& target_folder = ws->RootFolder();
      // We will write each individual tensor to its individual file.
      log_file_.reset(new std::ofstream(
          target_folder + "/" + def().input(0) + kPrintFileExtension,
          std::ofstream::out | std::ofstream::trunc));
      CHECK(log_file_->good()) << "Failed to open PrintOp file for tensor "
                               << def().input(0)
                               << ". rdstate() = " << log_file_->rdstate();
    }
  }

  ~PrintOp() {
    if (log_file_.get()) {
      log_file_->close();
    }
  }

  bool RunOnDevice() override {
    // special-case empty tensors since they may have no meta()
    if (Input(0).size() == 0) {
      if (to_file_) {
        (*log_file_) << std::endl;
      } else {
        LOG(INFO) << MetaStr();
      }
      return true;
    }

    if (OperatorBase::InputIsType<TensorCPU>(0)) {
      return DispatchHelper<
          TensorTypes<float, double, int, long, bool, std::string>>::call(
              this, OperatorBase::Input<TensorCPU>(0));
    } else {
      return DispatchHelper<TensorTypes<float, double, int, long, bool>>::call(
          this, Input(0));
    }
  }

 private:
  std::string MetaStr() {
    std::stringstream meta_stream;
    meta_stream << "Tensor " << def().input(0) << " (";
    for (const auto dim : Input(0).dims()) {
      meta_stream << dim << ",";
    }
    meta_stream << "): ";
    return meta_stream.str();
  }

  template <typename T>
  bool DoRunWithType() {
    // A simple strategy to copy tensor if needed, and have the tensor pointer
    // pointing to the right instantiation. Note that tensor_copy_if_needed
    // will handle memory deallocation itself so no smart pointer is needed.
    const TensorCPU* tensor;
    TensorCPU tensor_copy_if_needed;
    if (OperatorBase::InputIsType<TensorCPU>(0)) {
      tensor = &OperatorBase::Input<TensorCPU>(0);
    } else {
      tensor_copy_if_needed.CopyFrom(Input(0), &context_);
      // Make sure that the copy is finished.
      context_.FinishDeviceComputation();
      tensor = &tensor_copy_if_needed;
    }
    std::stringstream values_stream;
    // One most likely doesn't want to print int64-number of items for visual
    // inspection, so we cast down to int here.
    int total_count = std::min(tensor->size(), TIndex(limit_));
    const T* tensor_data = tensor->template data<T>();
    for (int i = 0; i < total_count - 1; ++i) {
      values_stream << tensor_data[i] << ",";
    }
    // We do not add a comma after the last item.
    values_stream << tensor_data[total_count - 1];
    if (to_file_) {
      (*log_file_) << values_stream.str() << std::endl;
    } else {
      // Log to console.
      LOG(INFO) << MetaStr() << values_stream.str();
    }
    return true;
  }

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
};

/**
 * @brief Alias op makes the output and the input share the same underlying
 * storage.
 *
 * WARNING: in general, in caffe2's operator interface different tensors should
 * have different underlying storage, which is the assumption made by
 * components such as the dependency engine and memory optimization. Thus, in
 * normal situations you should not use the AliasOp, especially in a normal
 * forward-backward pass.
 *
 * The Alias op is provided so one can achieve true asynchrony, such as
 * Hogwild, in a graph. But make sure you understand all the implications
 * similar to multi-thread computation before you use it explicitly.
 */
template <class Context>
class AliasOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AliasOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    DCHECK_GT(input.size(), 0);
    Output(0)->ResizeLike(input);
    Output(0)->ShareData(input);
    return true;
  }
};

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FlattenOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    DCHECK_GT(input.size(), 0);
    output->Resize(vector<TIndex>{input.dim(0), input.size() / input.dim(0)});
    context_.template CopyBytes<Context, Context>(
        input.nbytes(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

// Output gets the data of input(0), but reshapes it like input(1).
template <class Context>
class ResizeLikeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ResizeLikeOp);

  bool RunOnDevice() override {
    auto& input0 = Input(0);
    auto& input1 = Input(1);
    auto* output = Output(0);
    DCHECK_EQ(input0.size(), input1.size());
    output->ResizeLike(Input(1));
    context_.template CopyBytes<Context, Context>(
        input0.nbytes(),
        input0.raw_data(),
        output->raw_mutable_data(input0.meta()));
    return true;
  }
};

template <typename T, class Context>
class SumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumOp);

  bool RunOnDevice() override {
    auto& input0 = Input(0);
    auto* output = Output(0);
    if (InputSize() == 1) {
      output->CopyFrom(input0, &context_);
      return true;
    }
    output->ResizeLike(input0);
    T* output_data = output->template mutable_data<T>();
    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      CHECK(output->dims() == Input(i).dims())
          << ProtoDebugString(def()) << "\n"
          << output->dims() << "\n"
          << "Input " << i << ": " << Input(i).dims();
    }

    // Add the first two - works if in-place or not.
    math::Add(
        output->size(),
        input0.template data<T>(),
        Input(1).template data<T>(),
        output_data,
        &context_);
    // Add remaining.
    for (int i = 2; i < InputSize(); ++i) {
      math::Add(
          output->size(),
          output_data,
          Input(i).template data<T>(),
          output_data,
          &context_);
    }
    return true;
  }
};

// WeightedSumOp computes the weighted sum of several tensors. The input should
// be in the form X_0, weight_0, X_1, weight_1, ... where X_i all have the same
// shape, and weight_i are size 1 tensors that specifies the weight of each
// vector. Note that if one wants to do in-place computation, it could only be
// done with X_0 also as the output, but not other X_i.
template <typename T, class Context>
class WeightedSumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(WeightedSumOp);

  bool RunOnDevice() override {
    DCHECK_EQ(InputSize() % 2, 0);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    DCHECK_GT(X0.size(), 0);
    DCHECK_EQ(weight0.size(), 1);
    int size = X0.size();
    auto* output = Output(0);
    output->ResizeLike(X0);
    math::Scale<T, Context>(
        size,
        weight0.template data<T>(),
        X0.template data<T>(),
        output->template mutable_data<T>(),
        &context_);
    for (int i = 2; i < InputSize(); i += 2) {
      auto& X = Input(i);
      // Do a check: if the input is the same as output, we have a problem -
      // in-place update should always only happen with the zeroth input.
      if (&X == output) {
        LOG(ERROR) << "Input #" << i << " is the same as output. "
                   << "If you want to do in-place updates, put the output as "
                   << "input #0.";
        return false;
      }
      auto& weight = Input(i + 1);
      DCHECK_EQ(X.size(), size);
      DCHECK_EQ(weight.size(), 1);
      math::Axpy<T, Context>(
          size,
          weight.template data<T>(),
          X.template data<T>(),
          output->template mutable_data<T>(),
          &context_);
    }
    return true;
  }
};

/**
 * @brief Update slices of the tensor in-place with weighted sum.
 *
 * ScatterWeightedSumOp is similar to WeightedSum and computes the weighted sum
 * of several tensors. The first tensor has to be in-place and only slices of it
 * on the first dimension as indexed by INDICES will be updated.
 *
 * Input:
 *   X_0 - tensor to be updated
 *   weight_0 - scalar weight for X_0, applied only to slices affected,
 *   INDICES - 1-D list of indices on the first dimension of X_0 that need to be
 * updated
 *   X_1 - update slices, has to have shape of len(INDICES) + shape(X_0)[1:]
 *   weight_1 - scalar weight for X_1 update
 *   X_2, weight_2, ...
 *
 * Output:
 *   X_0 - has to be exactly the same tensor as the input 0
 *
 * Note: The op pretty much ignores the exact shapes of the input arguments and
 * cares only about sizes. It's done for performance consideration to avoid
 * unnecessary reshapes. Only first dimension of X_0 is important, let's call it
 * N. If M is the total size of X_0 and K is the size of INDICES then X_i is
 * assumed to be of shape K x (M / N) regardless of the real shape.
 *
 * Note: Each update in INDICES is applied independently which means that if
 * duplicated elements are present in INDICES the corresponding slice of X_0
 * will be scaled multiple times. Manual collapsing of INDICES is required
 * beforehand if necessary.
 *
 * Note: Updates are applied sequentially by inputs which might have undesired
 * consequences if the input tensor is accessed concurrently by different op
 * (e.g. when doing Hogwild). Other threads might see intermediate results even
 * on individual slice level, e.g. X_0 scaled by weight_0 but without any
 * updates applied.
 *
 * For now really works only on CPU because of INDICES access
 */
template <typename T, class Context>
class ScatterWeightedSumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ScatterWeightedSumOp);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
  }

 private:
  template <typename Index>
  bool DoRunWithType() {
    TIndex block_size = Input(0).size_from_dim(1);
    return DispatchHelper<FixedSizes<1>, Index>::call(this, block_size);
  }

  template <typename Index, int FixedSize>
  bool DoRunWithSize() {
    DCHECK_EQ(InputSize() % 2, 1);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    auto& indices = Input(2);
    auto* output = Output(0);
    CHECK_EQ(&X0, output) << "In place operation is required";

    DCHECK_GT(X0.size(), 0);
    DCHECK_GT(X0.ndim(), 0) << "X0 has to be at least the vector";
    DCHECK_EQ(weight0.size(), 1);
    TIndex M = X0.size();
    TIndex N = X0.dim(0);
    TIndex K = indices.size();
    TIndex block_size = M / N;
    T* data = output->template mutable_data<T>();
    const Index* idxs = indices.template data<Index>();
    T w0 = *weight0.template data<T>();
    // It's most likely a constant so exact comparison is fine
    if (w0 != 1.0) {
      for (int i = 0; i < K; ++i) {
        Index idx = idxs[i];
        DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                    << ", range 0 to " << N;
        math::Scale<T, Context, FixedSize>(
            block_size,
            w0,
            data + block_size * idx,
            data + block_size * idx,
            &context_);
      }
    }
    for (int inp = 3; inp < InputSize(); inp += 2) {
      auto& X = Input(inp);
      auto& weight = Input(inp + 1);
      DCHECK_EQ(X.size(), block_size * K);
      DCHECK_EQ(weight.size(), 1);
      const T* x_data = X.template data<T>();
      T w = *weight.template data<T>();
      for (int i = 0; i < K; ++i) {
        Index idx = idxs[i];
        // double-checking the indices, but it's fine as it's DCHECK only
        DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                    << ", range 0 to " << N;
        math::Axpy<T, Context, FixedSize>(
            block_size,
            w,
            x_data + block_size * i,
            data + block_size * idx,
            &context_);
      }
    }
    return true;
  }
};

/**
 * @brief Update slices of the tensor in-place by overriding.
 *
 * Input:
 *   DATA - tensor to be updated
 *   INDICES - 1-D list of indices on the first dimension of X_0 that need to be
 *             updated
 *   SLICES - update slices, has to have shape of len(INDICES) + shape(X_0)[1:]
 *
 * Output:
 *   DATA - has to be exactly the same tensor as the input 0
 *
 * Note: The op pretty much ignores the exact shapes of the input arguments and
 * cares only about sizes. It's done for performance consideration to avoid
 * unnecessary reshapes. Only first dimension of X_0 is important, let's call it
 * N. If M is the total size of X_0 and K is the size of INDICES then X_i is
 * assumed to be of shape K x (M / N) regardless of the real shape.
 *
 * Note: Each update in INDICES is applied independently which means that if
 * duplicated elements are present in INDICES arbitrary one will win.
 *
 * For now really works only on CPU because of INDICES access
 */
template <typename T, class Context>
class ScatterAssignOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ScatterAssignOp);

  bool RunOnDevice() override {
    // Use run-time polymorphism
    auto& indices = Input(INDICES);
    if (indices.template IsType<int32_t>()) {
      DoRun<int32_t>();
    } else if (indices.template IsType<int64_t>()) {
      DoRun<int64_t>();
    } else {
      LOG(FATAL) << "Unsupported type of INDICES in ScatterAssignOp: "
                 << indices.meta().name();
    }
    return true;
  }

 private:
  template <typename Index>
  void DoRun() {
    auto& input = Input(DATA);
    auto& indices = Input(INDICES);
    auto& slices = Input(SLICES);
    auto* output = Output(0);
    CHECK_EQ(&input, output) << "In place operation is required";

    DCHECK_GT(input.ndim(), 0) << "X0 has to be at least the vector";
    TIndex M = input.size();
    TIndex N = input.dim(0);
    TIndex K = indices.size();
    TIndex block_size = M / N;
    DCHECK_EQ(slices.size(), block_size * K);
    // TODO(dzhulgakov): it can be made to work with arbitrary data type by
    // using raw_mutable_data
    T* data = output->template mutable_data<T>();
    const Index* idxs = indices.template data<Index>();
    const T* slicesData = slices.template data<T>();
#pragma omp parallel for
    for (int i = 0; i < K; ++i) {
      Index idx = idxs[i];
      // double-checking the indices, but it's fine as it's DCHECK only
      DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                  << ", range 0 to " << N;
      context_.template Copy<T, Context, Context>(
          block_size, slicesData + block_size * i, data + block_size * idx);
    }
  }

  INPUT_TAGS(DATA, INDICES, SLICES);
};

template <class Context, class DstContext, class SrcContext>
class CopyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(CopyOp);

  bool RunOnDevice() override {
    auto& input = OperatorBase::Input<Tensor<SrcContext>>(0);
    auto* output = OperatorBase::Output<Tensor<DstContext>>(0);
    output->ResizeLike(input);
    this->context_.template CopyBytes<SrcContext, DstContext>(
        input.nbytes(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

template <class Context>
class LengthsToSegmentIdsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsToSegmentIdsOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& input = Input(0);
    auto* output = Output(0);
    auto* input_data = input.template data<Index>();

    CHECK_EQ(input.dims().size(), 1) << "Input must be a vector.";
    auto total_length =
        std::accumulate(input_data, input_data + input.size(), 0);

    output->Resize(total_length);
    auto* output_data = output->template mutable_data<int32_t>();

    int pos = 0;
    for (int i = 0; i < input.size(); ++i) {
      auto len = input_data[i];
      std::fill(output_data, output_data + len, i);
      output_data += len;
    }
    return true;
  }
};

template <class Context>
class SegmentIdsToLengthsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SegmentIdsToLengthsOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& input = Input(0);
    CHECK_EQ(input.dims().size(), 1) << "Input must be a vector.";
    auto* input_data = input.template data<Index>();
    auto input_size = input.size();
    auto* output = Output(0);
    // segment id starts from 0
    auto num_segments = input_size ? input_data[input_size - 1] + 1 : 0;
    output->Resize(num_segments);
    auto* output_data = output->template mutable_data<int64_t>();
    std::fill(output_data, output_data + num_segments, 0);
    for (int64_t i = 0; i < input_size; i++) {
      output_data[input_data[i]] += 1;
    }

    return true;
  }
};

template <class SIndex, class Context>
class SliceOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SliceOp);

  bool RunOnDevice() override {
    auto* output = Output(0);
    auto& data = Input(0);

    auto& starts = Input(1);
    auto& ends = Input(2);
    auto* starts_data = starts.template data<SIndex>();
    auto* ends_data = ends.template data<SIndex>();

    CHECK_EQ(starts.ndim(), 1);
    CHECK_EQ(ends.ndim(), 1);
    CHECK_LE(data.ndim(), starts.size());
    CHECK_EQ(starts.size(), ends.size());

    std::vector<SIndex> starts_idx(data.ndim());
    std::vector<SIndex> ends_idx(data.ndim());
    std::vector<SIndex> dst_sizes(data.ndim());

    for (int i = 0; i < data.ndim(); ++i) {
      if (i >= starts.size()) {
        starts_idx[i] = 0;
        ends_idx[i] = data.dims()[i];
        continue;
      }
      auto start = starts_data[i];
      auto end = ends_data[i];
      if (start < 0) {
        start = data.dims()[i] + 1 + start;
      }
      if (end < 0) {
        end = data.dims()[i] + 1 + end;
      }
      CHECK_GE(start, 0);
      CHECK_GE(end, 0);
      CHECK_LT(start, data.dims()[i]);
      CHECK_LE(end, data.dims()[i]);
      CHECK_GE(end, start);
      starts_idx[i] = start;
      ends_idx[i] = end;
      dst_sizes[i] = end - start;
    }
    // for now only supports slicing in 1 dimension
    int dim = -1;
    for (int i = 0; i < data.ndim(); ++i) {
      if (starts_idx[i] > 0 || ends_idx[i] < data.dims()[i]) {
        CHECK_EQ(dim, -1) << "Currently only possible to slice in 1 dimension.";
        dim = i;
      }
    }
    if (dim == -1) {
      output->CopyFrom(data, &context_);
      return true;
    }
    auto unit = std::accumulate(
        data.dims().begin() + dim + 1,
        data.dims().end(),
        1,
        std::multiplies<SIndex>());
    auto num_blocks = std::accumulate(
        data.dims().begin(),
        data.dims().begin() + dim,
        1,
        std::multiplies<SIndex>());
    output->Resize(dst_sizes);
    auto* src_bytes = (char*)data.raw_data();
    auto* dst_bytes = (char*)output->raw_mutable_data(data.meta());

    auto src_nbytes = data.nbytes();
    auto dst_nbytes = output->nbytes();

    auto src_block_size = unit * data.dims()[dim];
    auto dst_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
    auto src_offset = unit * starts_idx[dim];

    if (num_blocks == 0 || dst_block_size == 0) {
      return true;
    }

    if (data.meta().copy()) {
      CHECK(false) << "Complex types not supported yet.";
    } else {
      auto itemsize = data.meta().itemsize();
      auto src_block_size_bytes = itemsize * src_block_size;
      auto dst_block_size_bytes = itemsize * dst_block_size;
      auto src_offset_bytes = src_bytes + itemsize * src_offset;
      auto dst_offset_bytes = dst_bytes;
      for (int i = 0; i < num_blocks; ++i) {
        DCHECK_LE(
            src_offset_bytes + dst_block_size_bytes, src_bytes + src_nbytes);
        DCHECK_LE(
            dst_offset_bytes + dst_block_size_bytes, dst_bytes + dst_nbytes);
        this->context_.template CopyBytes<Context, Context>(
            dst_block_size_bytes,
            (void*)src_offset_bytes,
            (void*)dst_offset_bytes);
        src_offset_bytes += src_block_size_bytes;
        dst_offset_bytes += dst_block_size_bytes;
      }
    }
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(SliceOp);
};

template <class Context>
class HasElementsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(HasElementsOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<TensorCPU>(0);
    output->Resize(std::vector<TIndex>{});
    *output->template mutable_data<bool>() = input.size() > 0;
    return true;
  }
};

// RecordShapeOp records the shape of the input tensor to a vector of int. You
// mostly don't need this operator explicitly, and it is mostly used in the
// autodiff process.
template <class Context>
class ShapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ShapeOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<TensorCPU>(0);
    output->Resize(input.ndim());
    TIndex* output_data = output->template mutable_data<TIndex>();
    for (int i = 0; i < input.ndim(); ++i) {
      output_data[i] = input.dim(i);
    }
    return true;
  }
};

template <class Context>
class SqueezeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SqueezeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        dims_(OperatorBase::GetRepeatedArgument<int>("dims")) {
    auto originalSize = dims_.size();
    std::sort(dims_.begin(), dims_.end());
    std::unique(dims_.begin(), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CHECK(dims_.empty() || dims_.front() >= 0)
        << "Dimension ids must be non-negative.";
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    output->CopyFrom(input, &context_);

    if (dims_.empty()) {
      return true;
    }
    CHECK_GE(input.dims().back() + 1, dims_.size())
        << "Input needs at least " << (dims_.back() + 1) << " dimensions.";

    int j = 0;
    std::vector<int> newDims;
    for (int i = 0; i < input.dims().size(); ++i) {
      if (j < dims_.size() && dims_[j] == i) {
        CHECK_EQ(input.dims()[i], 1) << "Dimension " << i
                                     << " of input must be 1.";
        ++j;
        continue;
      }
      newDims.push_back(input.dims().at(i));
    }
    output->Reshape(newDims);
    return true;
  }

 private:
  vector<int> dims_;

 public:
  DISABLE_COPY_AND_ASSIGN(SqueezeOp);
};

template <class Context>
class ExpandDimsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ExpandDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        dims_(OperatorBase::GetRepeatedArgument<int>("dims")) {
    auto originalSize = dims_.size();
    std::sort(dims_.begin(), dims_.end());
    std::unique(dims_.begin(), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CHECK(dims_.empty() || dims_.front() >= 0)
        << "Dimension ids must be non-negative.";
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    output->CopyFrom(input, &context_);
    if (dims_.empty()) {
      return true;
    }

    auto newDims = input.dims();
    CHECK_GE(input.dims().size() + dims_.size(), dims_.back() + 1)
        << "Input needs at least " << (1 + dims_.back() - dims_.size())
        << " dimensions given `dims`.";
    for (const auto dim : dims_) {
      newDims.insert(newDims.begin() + dim, 1);
    }
    output->Reshape(newDims);
    return true;
  }

 private:
  vector<int> dims_;
};

// Since we just do copying, consider untemplating it on T and using raw_data()
template <typename T, class Context>
class GatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(GatherOp);

  bool RunOnDevice() override {
    // Use run-time polymorphism
    auto& indices = Input(INDICES);
    if (indices.template IsType<int32_t>()) {
      DoRun<int32_t>();
    } else if (indices.template IsType<int64_t>()) {
      DoRun<int64_t>();
    } else {
      LOG(FATAL) << "Unsupported type of INDICES in Gather: "
                 << indices.meta().name();
    }
    return true;
  }

 private:
  template <typename Index>
  void DoRun() {
    // If we endup using it on GPU doint O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CHECK_GE(data.ndim(), 1) << "DATA should be at least 1-D";
    auto shape = indices.dims();
    shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
    output->Resize(shape);

    int block_size = data.size() / data.dim(0);
    int N = indices.size();

    const T* d = data.template data<T>();
    const Index* idxs = indices.template data<Index>();
    T* out = output->template mutable_data<T>();

    for (int i = 0; i < N; ++i) {
      context_.template Copy<T, Context, Context>(
          block_size, d + block_size * idxs[i], out + block_size * i);
    }
  }

 public:
  INPUT_TAGS(DATA, INDICES);
};

// Since we just do copying, consider untemplating it on T and using raw_data()
/**
 * Deduplicates input indices vector and optionally produces reverse remapping.
 * Current implementation produces a sorted list but it's not guaranteed in
 * general.
 */
template <class Context>
class UniqueOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(UniqueOp);

  bool RunOnDevice() override {
    // Use run-time polymorphism
    auto& input = Input(0);
    if (input.template IsType<int32_t>()) {
      DoRun<int32_t>();
    } else if (input.template IsType<int64_t>()) {
      DoRun<int64_t>();
    } else {
      LOG(FATAL) << "Unsupported type of input in Unique: "
                 << input.meta().name();
    }
    return true;
  }

 private:
  vector<int> order_;

  template <typename T>
  void DoRun() {
    auto& inputTensor = Input(0);
    // use dim32 to enforce that it's fine to have remapping of type int
    int N = inputTensor.dim32(0);
    CHECK_EQ(inputTensor.ndim(), 1) << "Input should be a vector";
    auto* uniqueTensor = Output(UNIQUE);

    int* remapping = nullptr;
    if (REMAPPING < OutputSize()) {
      auto* remappingTensor = Output(REMAPPING);
      remappingTensor->ResizeLike(inputTensor);
      remapping = remappingTensor->template mutable_data<int>();
    }

    const T* input = inputTensor.template data<T>();
    // TODO(dzhulgakov): if perf becomes an issue consider doing hash table
    // instead of sorting
    order_.resize(N);
    std::iota(order_.begin(), order_.end(), 0);
    std::sort(order_.begin(), order_.end(), [input](const int x, const int y) {
      return input[x] < input[y];
    });
    int K = N;
    for (int i = 1; i < N; ++i) {
      K -= input[order_[i]] == input[order_[i - 1]];
    }
    uniqueTensor->Resize(K);
    T* unique = uniqueTensor->template mutable_data<T>();
    K = 0;
    T prev = -1;
    for (int i = 0; i < N; ++i) {
      if (i == 0 || prev != input[order_[i]]) {
        prev = unique[K++] = input[order_[i]];
      }
      if (remapping) {
        remapping[order_[i]] = K - 1;
      }
    }
  }

 public:
  OUTPUT_TAGS(UNIQUE, REMAPPING);
};
} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILITY_OPS_H_

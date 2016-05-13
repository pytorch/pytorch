#ifndef CAFFE2_OPERATORS_UTILITY_OPS_H_
#define CAFFE2_OPERATORS_UTILITY_OPS_H_

#include <fstream>
#include <sstream>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

const char kPrintFileExtension[] = ".log";

// FreeOp frees the content of the output blob. We allow it to take in input
// blobs purely for the reason that it can "wait" on the input blobs to be
// produced by some of the earlier operators before a free is called.
class FreeOp : public OperatorBase {
 public:
  USE_SIMPLE_BASE_CTOR_DTOR(FreeOp);

  bool Run() override {
    for (Blob* output : Outputs()) {
      output->Reset();
    }
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(FreeOp);
};

template <typename T, class Context>
class PrintOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
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
      log_files_.resize(def().input_size());
      for (int i = 0; i < def().input_size(); ++i) {
        log_files_[i].reset(new std::ofstream(
            target_folder + "/" + def().input(i) + kPrintFileExtension,
            std::ofstream::out | std::ofstream::trunc));
        CAFFE_CHECK(log_files_[i]->good())
            << "Failed to open PrintOp file for tensor " << def().input(i)
            << ". rdstate() = " << log_files_[i]->rdstate();
      }
    }
  }

  ~PrintOp() {
    for (auto& log_file : log_files_) {
      log_file->close();
    }
  }

  bool RunOnDevice() override {
    TensorCPU temp_tensor;
    for (int input_id = 0; input_id < InputSize(); ++input_id) {
      // A special case for inputs that are on CPUContext: in which case we
      // would not need to do any copy.
      if (OperatorBase::InputIsType<TensorCPU>(input_id)) {
        auto& input = OperatorBase::Input<TensorCPU>(input_id);
        temp_tensor.ReshapeLike(input);
        temp_tensor.ShareData(input);
      } else {
        auto& input = Input(input_id);
        CAFFE_DCHECK_GT(input.size(), 0);
        temp_tensor.ReshapeLike(input);
        context_.template Copy<T, Context, CPUContext>(
            input.size(), input.template data<T>(),
            temp_tensor.template mutable_data<T>());
      }
      std::stringstream values_stream;
      // One most likely doesn't want to print int64-number of items for visual
      // inspection, so we cast down to int here.
      int total_count = std::min(int(temp_tensor.size()), limit_);
      const T* temp_tensor_data = temp_tensor.data<T>();
      for (int i = 0; i < total_count - 1; ++i) {
        values_stream << temp_tensor_data[i] << ",";
      }
      // We do not add a comma after the last item.
      values_stream << temp_tensor_data[total_count - 1];
      if (to_file_) {
        // Also log to file.
        auto& stream = *log_files_[input_id];
        stream << values_stream.str() << std::endl;
      } else {
        std::stringstream dims_stream;
        for (const int dim : temp_tensor.dims()) {
          dims_stream << dim << ",";
        }
        // Log to console.
        CAFFE_LOG_INFO << "Tensor " << def().input(input_id)
            << " (" << dims_stream.str() << "): " << values_stream.str();
      }
    }
    return true;
  }

 private:
  bool to_file_;
  int limit_;
  vector<std::unique_ptr<std::ofstream> > log_files_;
  DISABLE_COPY_AND_ASSIGN(PrintOp);
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
    CAFFE_DCHECK_GT(input.size(), 0);
    Output(0)->ReshapeLike(input);
    Output(0)->ShareData(input);
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(AliasOp);
};

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FlattenOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    CAFFE_DCHECK_GT(input.size(), 0);
    output->Reshape(
        vector<TIndex>{input.dim(0), input.size() / input.dim(0)});
    context_.template Memcpy<Context, Context>(
        input.nbytes(), input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(FlattenOp);
};

// Output gets the data of input(0), but reshapes it like input(1).
template <class Context>
class ReshapeLikeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ReshapeLikeOp);

  bool RunOnDevice() override {
    auto& input0 = Input(0);
    auto& input1 = Input(1);
    auto* output = Output(0);
    CAFFE_DCHECK_EQ(input0.size(), input1.size());
    output->ReshapeLike(Input(1));
    context_.template Memcpy<Context, Context>(
        input0.nbytes(), input0.raw_data(),
        output->raw_mutable_data(input0.meta()));
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(ReshapeLikeOp);
};

template <class Context>
class SplitOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SplitOp);

  bool RunOnDevice() override {
    const auto& input = Input(0);
    for (int i = 0; i < OutputSize(); ++i) {
      auto* output = Output(i);
      output->ReshapeLike(input);
      output->ShareData(input);
    }
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(SplitOp);
};

template <typename T, class Context>
class SumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumOp);

  bool RunOnDevice() override {
    auto& input0 = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input0);
    T* output_data = output->template mutable_data<T>();
    if (InputSize() == 1) {
      if (input0.template data<T>() == output_data) {
        // in-place, single element, so nothing to do
        return true;
      }
      // Otherwise, copy input0 into output
      context_.template Copy<T, Context, Context>(
          input0.size(), input0.template data<T>(),
          output_data);
      return true;
    }
    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      CAFFE_CHECK(output->dims() == Input(i).dims())
          << ProtoDebugString(def()) << "\n"
          << output->dims() << "\n"
          << "Input " << i << ": " << Input(i).dims();
    }

    // Add the first two - works if in-place or not.
    math::Add(output->size(), input0.template data<T>(),
              Input(1).template data<T>(),
              output_data, &context_);
    // Add remaining.
    for (int i = 2; i < InputSize(); ++i) {
      math::Add(output->size(), output_data, Input(i).template data<T>(),
                output_data, &context_);
    }
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(SumOp);
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
    CAFFE_DCHECK_EQ(InputSize() % 2, 0);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    CAFFE_DCHECK_GT(X0.size(), 0);
    CAFFE_DCHECK_EQ(weight0.size(), 1);
    int size = X0.size();
    auto* output = Output(0);
    output->ReshapeLike(X0);
    math::Scale<T, Context>(
        size, weight0.template data<T>(), X0.template data<T>(),
        output->template mutable_data<T>(),
        &context_);
    for (int i = 2; i < InputSize(); i += 2) {
      auto& X = Input(i);
      // Do a check: if the input is the same as output, we have a problem -
      // in-place update should always only happen with the zeroth input.
      if (&X == output) {
        CAFFE_LOG_ERROR << "Input #" << i << " is the same as output. "
                   << "If you want to do in-place updates, put the output as "
                   << "input #0.";
        return false;
      }
      auto& weight = Input(i + 1);
      CAFFE_DCHECK_EQ(X.size(), size);
      CAFFE_DCHECK_EQ(weight.size(), 1);
      math::Axpy<T, Context>(
          size, weight.template data<T>(), X.template data<T>(),
          output->template mutable_data<T>(),
          &context_);
    }
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(WeightedSumOp);
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

  bool RunOnDevice() override {
    // Use run-time polymorphism
    auto& indices = Input(2);
    if (indices.template IsType<int32_t>()) {
      DoRun<int32_t>();
    } else if (indices.template IsType<int64_t>()) {
      DoRun<int64_t>();
    } else {
      CAFFE_LOG_FATAL << "Unsupported type of INDICES in ScatterWeightdSumOp: "
                      << indices.meta().name();
    }
    return true;
  }

 private:
  template <typename Index>
  void DoRun() {
    CAFFE_DCHECK_EQ(InputSize() % 2, 1);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    auto& indices = Input(2);
    auto* output = Output(0);
    CAFFE_CHECK_EQ(&X0, output) << "In place operation is required";

    CAFFE_DCHECK_GT(X0.size(), 0);
    CAFFE_DCHECK_GT(X0.ndim(), 0) << "X0 has to be at least the vector";
    CAFFE_DCHECK_EQ(weight0.size(), 1);
    int M = X0.size();
    int N = X0.dim(0);
    int K = indices.size();
    int block_size = M / N;
    T* data = output->template mutable_data<T>();
    const Index* idxs = indices.template data<Index>();
    T w0 = *weight0.template data<T>();
    // It's most likely a constant so exact comparison is fine
    if (w0 != 1.0) {
      for (int i = 0; i < K; ++i) {
        Index idx = idxs[i];
        CAFFE_DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                          << ", range 0 to " << N;
        math::Scale<T, Context>(
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
      CAFFE_DCHECK_EQ(X.size(), block_size * K);
      CAFFE_DCHECK_EQ(weight.size(), 1);
      const T* x_data = X.template data<T>();
      T w = *weight.template data<T>();
      for (int i = 0; i < K; ++i) {
        Index idx = idxs[i];
        // double-checking the indices, but it's fine as it's DCHECK only
        CAFFE_DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                          << ", range 0 to " << N;
        math::Axpy<T, Context>(
            block_size,
            w,
            x_data + block_size * i,
            data + block_size * idx,
            &context_);
      }
    }
  }

  DISABLE_COPY_AND_ASSIGN(ScatterWeightedSumOp);
};

template <class Context, class DstContext, class SrcContext>
class CopyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(CopyOp);

  bool RunOnDevice() override {
    auto& input = OperatorBase::Input<Tensor<SrcContext> >(0);
    auto* output = OperatorBase::Output<Tensor<DstContext> >(0);
    output->ReshapeLike(input);
    this->context_.template Memcpy<SrcContext, DstContext>(
      input.nbytes(),
      input.raw_data(),
      output->raw_mutable_data(input.meta()));
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(CopyOp);
};

// RecordShapeOp records the shape of the input tensor to a vector of int. You
// mostly don't need this operator explicitly, and it is mostly used in the
// autodiff process.
template <class Context>
class RecordShapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(RecordShapeOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<vector<TIndex> >(0);
    *output = input.dims();
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(RecordShapeOp);
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
      CAFFE_LOG_FATAL << "Unsupported type of INDICES in Gather: "
                      << indices.meta().name();
    }
    return true;
  }

 private:
  template <typename Index>
  bool DoRun() {
    // If we endup using it on GPU doint O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_CHECK_GE(data.ndim(), 1) << "DATA should be at least 1-D";
    auto shape = indices.dims();
    shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
    output->Reshape(shape);

    int block_size = data.size() / data.dim(0);
    int N = indices.size();

    const T* d = data.template data<T>();
    const Index* idxs = indices.template data<Index>();
    T* out = output->template mutable_data<T>();

    for (int i = 0; i < N; ++i) {
      context_.template Copy<T, Context, Context>(
          block_size, d + block_size * idxs[i], out + block_size * i);
    }
    return true;
  }

 public:
  INPUT_TAGS(DATA, INDICES);
  DISABLE_COPY_AND_ASSIGN(GatherOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_UTILITY_OPS_H_

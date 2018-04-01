#ifndef CAFFE2_OPERATORS_UTILITY_OPS_H_
#define CAFFE2_OPERATORS_UTILITY_OPS_H_

#include <math.h>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

#include <map>
#include <utility>

namespace caffe2 {

template <class Context>
class NanCheckOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  NanCheckOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;

 private:
  TensorPrinter tensorPrinter_;
  Tensor<Context> scratch_;
};

struct GetNanCheckGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return {CreateOperatorDef(
        "NanCheck",
        "",
        std::vector<string>{GO(0)},
        std::vector<string>{GI(0)})};
  }
};

template <class Context>
class WallClockTimeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  WallClockTimeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    int64_t nanoseconds = static_cast<long int>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count());

    TensorCPU* output = OperatorBase::Output<TensorCPU>(0);
    output->Resize();
    *output->template mutable_data<int64_t>() = nanoseconds;

    return true;
  }
};

const char kPrintFileExtension[] = ".log";

template <class Context>
class PrintOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;
  PrintOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tensor_printer_(
            operator_def.input(0),
            OperatorBase::GetSingleArgument<int>("to_file", 0)
                ? ws->RootFolder() + "/" + operator_def.input(0) +
                    kPrintFileExtension
                : "",
            OperatorBase::GetSingleArgument<int>("limit", 0)),
        every_n_(OperatorBase::GetSingleArgument<int>("every_n", 1)) {
    CAFFE_ENFORCE_GE(every_n_, 1);
  }

  bool RunOnDevice() override {
    if (++occurrences_mod_n_ > every_n_) {
      occurrences_mod_n_ -= every_n_;
    }
    if (occurrences_mod_n_ != 1) {
      return true;
    }

    if (!OperatorBase::InputIsType<Tensor<Context>>(0) &&
        !OperatorBase::InputIsType<TensorCPU>(0)) {
      LOG(INFO) << "Blob of type: "
                << OperatorBase::Inputs().at(0)->meta().name();
      return true;
    }
    // special-case empty tensors since they may have no meta()
    if (Input(0).size() == 0) {
      tensor_printer_.PrintMeta(Input(0));
      return true;
    }

    using Types = TensorTypes<
        float,
        double,
        int,
        long,
        bool,
        char,
        unsigned char,
        std::string>;

    if (OperatorBase::InputIsType<TensorCPU>(0)) {
      return DispatchHelper<Types>::call(
          this, OperatorBase::Input<TensorCPU>(0));
    } else {
      return DispatchHelper<Types>::call(this, Input(0));
    }
  }

 private:
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
    tensor_printer_.Print<T>(*tensor);
    return true;
  }

 private:
  TensorPrinter tensor_printer_;
  int every_n_;
  int occurrences_mod_n_{0};
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
    CAFFE_ENFORCE_GE(input.size(), 0, "Tensor is not initialized");
    Output(0)->ResizeLike(input);
    Output(0)->ShareData(input);
    return true;
  }
};

/**
 * @brief Pass inputs to outputs.
 * Input:
 *   DATA - dense tensor.
 * Output:
 *   DATA - same tensor as input.
 */
template <class Context>
class EnsureDenseOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(EnsureDenseOp)

  bool RunOnDevice() override {
    const auto& input = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE_GT(input.ndim(), 0, "Input has to be at least a vector.");
    // it is allowed to have the output inplace overwrite the input but also
    // allow the output to be copied from the input
    if (&input != output) {
      output->ResizeLike(input);
      output->CopyFrom(input, &context_);
    }
    return true;
  }
};

template <class Context>
class FlattenToVecOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FlattenToVecOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE_GE(
        input.dims().size(), 1, "The rank of the tensor must be >= 1.");
    output->Resize(input.size());

    context_.template CopyItems<Context, Context>(
        input.meta(),
        input.size(),
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
    CAFFE_ENFORCE_EQ(input0.size(), input1.size());
    output->ResizeLike(Input(1));
    context_.template CopyItems<Context, Context>(
        input0.meta(),
        input0.size(),
        input0.raw_data(),
        output->raw_mutable_data(input0.meta()));
    return true;
  }
};

template <class Context>
class SumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumOp);

  template <typename T, typename M>
  bool DoRunWithType() {
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
      if (output->dims() != Input(i).dims()) {
        CAFFE_THROW(
            "Check failed: output->dims() == Input(i).dims().",
            "Description: Input #",
            i,
            ", input dimension:",
            Input(i).dims(),
            " should match output dimension: ",
            output->dims());
      }
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

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float, float>();
    } else if (Input(0).template IsType<int>()) {
      return DoRunWithType<int, int>();
    } else {
      CAFFE_THROW(
          "Sum operator only supports 32-bit float and ints, but",
          " input was of type ",
          Input(0).meta().name());
    }
  }
};

// WeightedSumOp computes the weighted sum of several tensors. The input should
// be in the form X_0, weight_0, X_1, weight_1, ... where X_i all have the same
// shape, and weight_i are size 1 tensors that specifies the weight of each
// vector. Note that if one wants to do in-place computation, it could only be
// done with X_0 also as the output, but not other X_i.
template <class Context>
class WeightedSumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(WeightedSumOp);

  template <typename DstType>
  bool DoRunWithType() {
    CAFFE_ENFORCE_EQ(InputSize() % 2, 0);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    CAFFE_ENFORCE_GT(X0.size(), 0);
    CAFFE_ENFORCE_EQ(weight0.size(), 1);
    int size = X0.size();
    auto* output = Output(0);
    output->ResizeLike(X0);
    math::Scale<DstType, Context>(
        size,
        weight0.template data<float>(),
        X0.template data<DstType>(),
        output->template mutable_data<DstType>(),
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
      CAFFE_ENFORCE_EQ(X.size(), size);
      CAFFE_ENFORCE_EQ(weight.size(), 1);
      math::Axpy<DstType, Context>(
          size,
          weight.template data<float>(),
          X.template data<DstType>(),
          output->template mutable_data<DstType>(),
          &context_);
    }
    return true;
  }
  bool RunOnDevice() override;
};

template <class Context>
class WeightedSumGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  WeightedSumGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        grad_on_w_(OperatorBase::GetSingleArgument<bool>("grad_on_w", false)) {}

  template <typename DstType>
  bool DoRunWithType() {
    CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
    auto output_size = grad_on_w_ ? InputSize() - 1 : InputSize() / 2;
    CAFFE_ENFORCE_EQ(OutputSize(), output_size);

    auto& dY = Input(0);
    const auto* dY_data = dY.template data<DstType>();
    int size = dY.size();

    // The input size should be the input size of the forward op plus 1
    for (int i = 0; i < InputSize() / 2; i++) {
      auto& cur_w = Input(2 * i + 2);
      CAFFE_ENFORCE_EQ(cur_w.size(), 1);
      auto* cur_dX = Output(i);
      cur_dX->ResizeLike(dY);

      math::Scale<DstType, Context>(
          size,
          cur_w.template data<float>(),
          dY_data,
          cur_dX->template mutable_data<DstType>(),
          &context_);

      if (grad_on_w_) {
        auto& cur_X = Input(2 * i + 1);
        CAFFE_ENFORCE_EQ(cur_X.size(), size);
        auto* cur_dw = Output(i + output_size / 2);
        cur_dw->Resize(1);
        math::Dot<DstType, Context>(
            size,
            dY_data,
            cur_X.template data<DstType>(),
            cur_dw->template mutable_data<float>(),
            &context_);
      }
    }

    return true;
  }

  bool RunOnDevice() override;

 private:
  bool grad_on_w_;
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
    return DispatchHelper<FixedValues<1>, Index>::call(this, block_size);
  }

  template <typename Index, int FixedSize>
  bool DoRunWithValue() {
    CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    auto& indices = Input(2);
    auto* output = Output(0);
    CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");

    CAFFE_ENFORCE_GT(X0.size(), 0);
    CAFFE_ENFORCE_GT(X0.ndim(), 0, "X0 has to be at least the vector");
    CAFFE_ENFORCE_EQ(weight0.size(), 1);
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
        CAFFE_ENFORCE(
            0 <= idx && idx < N,
            "Index out of bounds: ",
            idx,
            ", range 0 to ",
            N);
        math::ScaleFixedSize<T, Context, FixedSize>(
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
      CAFFE_ENFORCE_EQ(X.size(), block_size * K);
      CAFFE_ENFORCE_EQ(weight.size(), 1);
      const T* x_data = X.template data<T>();
      T w = *weight.template data<T>();
      for (int i = 0; i < K; ++i) {
        Index idx = idxs[i];
        // double-checking the indices, but it's fine as it's DCHECK only
        DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                    << ", range 0 to " << N;
        math::AxpyFixedSize<T, Context, FixedSize>(
            block_size,
            w,
            x_data + block_size * i,
            data + block_size * idx,
            &context_);
      }
    }
    return true;
  }
  Tensor<CPUContext> x_data_host_;
  Tensor<CPUContext> weights_host_;
  Tensor<Context> x_data_device_;
  Tensor<Context> weights_device_;
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
template <class Context>
class ScatterAssignOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  virtual ~ScatterAssignOp() {}

  ScatterAssignOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        runners_({{{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT},
                   &ScatterAssignOp::DoRun<int32_t, float>},
                  {{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT16},
                   &ScatterAssignOp::DoRun<int32_t, float16>},
                  {{TensorProto_DataType_INT32, TensorProto_DataType_INT32},
                   &ScatterAssignOp::DoRun<int32_t, int32_t>},
                  {{TensorProto_DataType_INT32, TensorProto_DataType_INT64},
                   &ScatterAssignOp::DoRun<int32_t, int64_t>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT},
                   &ScatterAssignOp::DoRun<int64_t, float>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT16},
                   &ScatterAssignOp::DoRun<int64_t, float16>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_INT32},
                   &ScatterAssignOp::DoRun<int64_t, int32_t>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_INT64},
                   &ScatterAssignOp::DoRun<int64_t, int64_t>}}) {}

  bool RunOnDevice() override {
    const auto& data = Input(DATA);
    const auto& slices = Input(SLICES);
    auto& indices = Input(INDICES);

    const auto dataType = TypeMetaToDataType(data.meta());
    const auto slicesType = TypeMetaToDataType(slices.meta());
    const auto indicesType = TypeMetaToDataType(indices.meta());
    auto* output = Output(0);

    auto runner = GetRunner(dataType, slicesType, indicesType);
    (this->*runner)();
    return true;
  }

 private:
  typedef void (ScatterAssignOp::*RunnerType)();
  typedef std::
      map<std::pair<TensorProto_DataType, TensorProto_DataType>, RunnerType>
          RunnerMap;

  RunnerMap runners_;

  RunnerType GetRunner(
      const TensorProto_DataType dataType,
      const TensorProto_DataType slicesType,
      const TensorProto_DataType indicesType) {
    CAFFE_ENFORCE_EQ(dataType, slicesType, "Data and slice types must match");
    auto it = runners_.find({indicesType, dataType});
    CAFFE_ENFORCE(
        it != runners_.end(),
        "Could not find the runner corresponding to indicesType, dataType = ",
        indicesType,
        " ",
        dataType);
    return it->second;
  }

  template <typename Index, typename T>
  void DoRun() {
    auto& input = Input(DATA);
    auto& indices = Input(INDICES);
    auto& slices = Input(SLICES);
    auto* output = Output(0);
    CAFFE_ENFORCE_EQ(&input, output, "In place operation is required");

    CAFFE_ENFORCE_GT(input.ndim(), 0, "X0 has to be at least the vector");
    TIndex M = input.size();
    TIndex N = input.dim(0);
    TIndex K = indices.size();
    TIndex block_size = M / N;
    CAFFE_ENFORCE_EQ(slices.size(), block_size * K);
    // TODO(dzhulgakov): it can be made to work with arbitrary data type by
    // using raw_mutable_data
    T* data = output->template mutable_data<T>();
    const Index* idxs = indices.template data<Index>();
    const T* slicesData = slices.template data<T>();
    DoScatterAssign(data, idxs, slicesData, N, K, block_size);
  }

  template <typename Index, typename T>
  void DoScatterAssign(
      T* data,
      const Index* idxs,
      const T* slicesData,
      TIndex N,
      TIndex K,
      TIndex block_size) {
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
    this->context_.template CopyItems<SrcContext, DstContext>(
        input.meta(),
        input.size(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

template <class Context, class DstContext, class SrcContext>
class CopyOnDeviceLikeOp : public CopyOp<Context, DstContext, SrcContext> {
 public:
  CopyOnDeviceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : CopyOp<Context, DstContext, SrcContext>(operator_def, ws) {}
};

template <class Context>
class LengthsToSegmentIdsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsToSegmentIdsOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    auto* input_data = input.template data<int32_t>();

    CAFFE_ENFORCE(input.dims().size() == 1, "Input must be a vector.");
    auto total_length =
        std::accumulate(input_data, input_data + input.size(), 0);

    output->Resize(total_length);
    auto* output_data = output->template mutable_data<int32_t>();

    for (int i = 0; i < input.size(); ++i) {
      auto len = input_data[i];
      std::fill(output_data, output_data + len, i);
      output_data += len;
    }
    return true;
  }
};

template <class Context>
class LengthsToRangesOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsToRangesOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    auto* input_data = input.template data<int32_t>();

    CAFFE_ENFORCE(input.dims().size() == 1, "Input must be a vector.");
    auto size = input.size();

    output->Resize(size, 2);
    auto* output_data = output->template mutable_data<int32_t>();

    int32_t offset = 0;
    for (int i = 0; i < size; ++i) {
      auto len = input_data[i];
      output_data[i * 2] = offset;
      output_data[i * 2 + 1] = len;
      offset += len;
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
    if (input.ndim() == 2) {
      CAFFE_ENFORCE(
          input.dim32(0) == 1 || input.dim32(1) == 1,
          "Input must be a vector.");
    } else {
      CAFFE_ENFORCE_EQ(input.ndim(), 1, "Input must be a vector.");
    }
    auto* input_data = input.template data<Index>();
    auto input_size = input.size();
    auto* output = Output(0);
    // segment id starts from 0
    auto num_segments = input_size ? input_data[input_size - 1] + 1 : 0;
    if (InputSize() > 1) {
      CAFFE_ENFORCE_GE(Input(1).ndim(), 1);
      CAFFE_ENFORCE_LE(
          num_segments,
          Input(1).dim(0),
          "The number of segments inferred should *NOT* be larger "
          "than the size of Input(1)'s first dimension");
      num_segments = Input(1).dim(0);
    }
    CAFFE_ENFORCE(0 <= num_segments, "Indices must be in 0..K-1 range");
    output->Resize(num_segments);
    auto* output_data = output->template mutable_data<int32_t>();
    if (num_segments == 0) {
      return true;
    }
    std::fill(output_data, output_data + num_segments, 0);
    Index prev = 0; // Assume that segment_id >= 0.
    for (int64_t i = 0; i < input_size; i++) {
      CAFFE_ENFORCE(
          prev <= input_data[i],
          "Segment ids must be sorted: ",
          prev,
          " vs ",
          input_data[i]);
      prev = input_data[i];
      output_data[input_data[i]] += 1;
    }

    return true;
  }
};

template <class Context>
class SegmentIdsToRangesOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SegmentIdsToRangesOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& input = Input(0);
    CAFFE_ENFORCE(input.dims().size() == 1, "Input must be a vector.");
    auto* input_data = input.template data<Index>();
    auto input_size = input.size();
    auto* output = Output(0);
    // segment id starts from 0
    auto num_segments = input_size ? input_data[input_size - 1] + 1 : 0;
    if (InputSize() > 1) {
      CAFFE_ENFORCE_GE(Input(1).ndim(), 1);
      CAFFE_ENFORCE_LE(
          num_segments,
          Input(1).dim(0),
          "The number of segments inferred should *NOT* be larger "
          "than the size of Input(1)'s first dimension");
      num_segments = Input(1).dim(0);
    }
    CAFFE_ENFORCE(0 <= num_segments, "Indices must be in 0..K-1 range");
    output->Resize(num_segments, 2);
    auto* output_data = output->template mutable_data<int32_t>();
    if (num_segments == 0) {
      return true;
    }
    std::fill(output_data, output_data + num_segments * 2, 0);
    Index prev = input_data[0];
    for (int64_t i = 0; i < input_size; i++) {
      CAFFE_ENFORCE(
          prev <= input_data[i],
          "Segment ids must be sorted: ",
          prev,
          " vs ",
          input_data[i]);
      while (prev != input_data[i]) {
        ++prev;
        output_data[prev * 2] = i;
      }
      output_data[input_data[i] * 2 + 1] += 1;
    }

    return true;
  }
};

template <class Context>
class LengthsToWeightsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LengthsToWeightsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        power_(OperatorBase::GetSingleArgument<float>("power", 0.5)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& input = Input(0);
    CAFFE_ENFORCE(input.dims().size() == 1, "Input must be a vector.");
    auto* input_data = input.template data<Index>();
    auto input_size = input.size();
    auto* output = Output(0);

    int64_t output_size = 0;
    for (auto i = 0; i < input_size; i++) {
      CAFFE_ENFORCE_GE(input_data[i], 0, "unexpected negative length value");
      output_size += input_data[i];
    }

    std::function<float(const int64_t& length, const float& power)> getWeight;
    if (power_ == 0.5) {
      getWeight = [](const int64_t& length, const float& /*power*/) {
        return 1.0 / std::sqrt(length);
      };
    } else if (power_ == 1) {
      getWeight = [](const int64_t& length, const float& /*power*/) {
        return 1.0 / length;
      };
    } else {
      getWeight = [](const int64_t& length, const float& power) {
        return 1.0 / std::pow(length, power);
      };
    }

    output->Resize(output_size);
    auto* output_data = output->template mutable_data<float>();
    int64_t cnt = 0;
    for (auto i = 0; i < input_size; i++) {
      auto len = input_data[i];
      if (len == 0) {
        continue;
      }
      CAFFE_ENFORCE_LE(cnt + len, output_size, "unexpected lengths value");

      float weight_value = getWeight(len, power_);
      std::fill(output_data + cnt, output_data + cnt + len, weight_value);
      cnt += len;
    }

    return true;
  }

 private:
  float power_;
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

template <class Context>
class IsEmptyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(IsEmptyOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<TensorCPU>(0);
    output->Resize(std::vector<TIndex>{});
    *output->template mutable_data<bool>() = (input.size() == 0);
    return true;
  }
};

// Return the size of a tensor
template <class Context>
class SizeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SizeOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);

    output->Resize(vector<TIndex>());
    auto* output_data = output->template mutable_data<int64_t>();

    auto size = input.size();
    math::Set<int64_t, Context>(
        1, static_cast<int64_t>(size), output_data, &context_);

    return true;
  }
};

// returns a shape to be passed to Reshape
template <class Context>
class LengthsToShapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsToShapeOp);

  bool RunOnDevice() override {
    auto& input = Input(0);

    CAFFE_ENFORCE(input.dims().size() == 1, "Input must be a vector.");
    auto* output = Output(0);
    auto* input_data = input.template data<int32_t>();

    auto size = input.size();
    auto first = input_data[0];

    for (int i = 1; i < size; i++) {
      CAFFE_ENFORCE(
          input_data[i] == first, "All elements of input must be same ");
    }

    output->Resize(2);
    auto* output_data = output->template mutable_data<int32_t>();
    output_data[0] = size;
    output_data[1] = first;

    return true;
  }
};

template <class Context>
class GatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(GatherOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(INDICES));
  }

  template <typename Index>
  bool DoRunWithType() {
    // If we endup using it on GPU doing O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    auto shape = indices.dims();
    shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
    output->Resize(shape);

    int block_size = data.size_from_dim(1);
    auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
    int N = indices.size();

    auto src_base = static_cast<const char*>(data.raw_data());
    const Index* idxs = indices.template data<Index>();
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));

    for (int i = 0; i < N; ++i) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.dim(0),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.dim(0));
      auto src = src_base + idx * block_bytesize;
      context_.template CopyItems<Context, Context>(
          data.meta(), block_size, src, out + block_bytesize * i);
    }
    return true;
  }

  INPUT_TAGS(DATA, INDICES);
};

template <class Context>
class GatherRangesOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(GatherRangesOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(RANGES));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& ranges = Input(RANGES);
    auto* outputData = Output(0);
    auto* outputLengths = Output(1);

    auto batchSize = ranges.dim(0);
    CAFFE_ENFORCE(data.ndim() == 1, "Data has to be 1-D");
    CAFFE_ENFORCE(ranges.ndim() == 3, "Ranges must be 3-D");
    CAFFE_ENFORCE(ranges.dim(1) > 0, "There has to be at least one range");
    CAFFE_ENFORCE_EQ(
        ranges.dim(2), 2, "Ranges last dimention should be of size 2");

    auto* rawData = static_cast<const char*>(data.raw_data());
    auto* rangesData = ranges.template data<Index>();

    outputLengths->Resize(batchSize);
    auto* outputLengthsPtr = outputLengths->template mutable_data<int32_t>();
    size_t start = 0;
    size_t blockSize = ranges.size_from_dim(1);
    for (size_t i = 0; i < batchSize; ++i) {
      auto end = start + blockSize;
      outputLengthsPtr[i] = accumulate(rangesData, start, end);
      start = end;
    }

    size_t outputSize = accumulate(rangesData, 0, ranges.size());
    outputData->Resize(outputSize);

    auto outputRawData =
        static_cast<char*>(outputData->raw_mutable_data(data.meta()));
    VLOG(1) << "Copying data";
    size_t outputOffsetBytes = 0;
    auto itemsize = data.meta().itemsize();
    for (int i = 0; i < ranges.size(); i += 2) {
      auto rangeStart = rangesData[i];
      auto rangeLength = rangesData[i + 1];
      if (!rangeLength) {
        continue;
      }
      auto rangeSizeBytes = rangeLength * itemsize;
      CAFFE_ENFORCE(outputOffsetBytes < outputSize * itemsize);
      CAFFE_ENFORCE(rangeStart + rangeLength <= data.size());
      context_.template CopyItems<Context, Context>(
          data.meta(),
          rangeLength,
          rawData + rangeStart * itemsize,
          outputRawData + outputOffsetBytes);
      outputOffsetBytes += rangeSizeBytes;
    }
    CAFFE_ENFORCE(outputOffsetBytes == outputSize * itemsize);
    return true;
  }

  INPUT_TAGS(DATA, RANGES, LENGTHS);

 private:
  template <typename Index>
  size_t accumulate(Index* ranges, size_t start, size_t end) {
    size_t result = 0;
    for (int i = start + 1; i < end; i += 2) {
      result += ranges[i];
    }
    return result;
  }
};

template <class Context>
class LengthsGatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsGatherOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(INDICES));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& items = Input(ITEMS);
    auto& lengths = Input(LENGTHS);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(items.ndim(), 1, "ITEMS should be at least 1-D");
    CAFFE_ENFORCE_EQ(lengths.ndim(), 1, "LENGTHS should be 1-D");
    CAFFE_ENFORCE_EQ(indices.ndim(), 1, "INDICES should be 1-D");

    const auto* lengths_data = lengths.template data<int32_t>();
    const auto* indices_data = indices.template data<Index>();

    TIndex total_length = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      auto idx = indices_data[i];
      CAFFE_ENFORCE_LT(idx, lengths.size());
      total_length += lengths_data[idx];
    }
    auto shape = items.dims();
    shape[0] = total_length;
    output->Resize(shape);

    offsets_.clear();
    TIndex running_offset = 0;
    offsets_.reserve(lengths.size());
    for (size_t i = 0; i < lengths.size(); ++i) {
      offsets_.push_back(running_offset);
      running_offset += lengths_data[i];
    }
    CAFFE_ENFORCE_EQ(
        items.dim(0),
        running_offset,
        "LENGTHS must match the first dimension of ITEMS");

    auto src_base = static_cast<const char*>(items.raw_data());
    auto block_size = items.size_from_dim(1);
    auto block_bytesize = block_size * items.itemsize();
    auto out = static_cast<char*>(output->raw_mutable_data(items.meta()));

    for (size_t i = 0; i < indices.size(); ++i) {
      auto idx = indices_data[i];
      auto length = lengths_data[idx];
      context_.template CopyItems<Context, Context>(
          items.meta(),
          length * block_size,
          src_base + offsets_[idx] * block_bytesize,
          out);
      out += length * block_bytesize;
    }
    return true;
  }

  std::vector<TIndex> offsets_;

  INPUT_TAGS(ITEMS, LENGTHS, INDICES);
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
  Tensor<Context> thrust_unique_buffer_;
  Tensor<Context> cuda_order_buffer_;
  Tensor<Context> second_order_buffer_;

  template <typename T>
  void DoRun();

 public:
  OUTPUT_TAGS(UNIQUE, REMAPPING);
};

template <class Context>
class UnsafeCoalesceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;

  bool RunOnDevice() override {
    size_t coalesced_size = 0;
    for (int i = 0; i < InputSize(); ++i) {
      CAFFE_ENFORCE(
          !Input(i).meta().ctor(),
          "Must only coalesce fundamental types, error at input: ",
          i);
    }

    auto roundToAlignment = [](size_t bytes) -> size_t {
      return ((bytes + gCaffe2Alignment - 1) / gCaffe2Alignment) *
          gCaffe2Alignment;
    };

    for (int i = 0; i < InputSize(); ++i) {
      coalesced_size += roundToAlignment(Input(i).nbytes());
    }

    auto* coalesced = Output(OutputSize() - 1);
    coalesced->Resize(coalesced_size);
    math::Set<uint8_t, Context>(
        coalesced_size,
        0.0,
        coalesced->template mutable_data<uint8_t>(),
        &context_);

    size_t coalesced_offset = 0;
    for (auto i = 0; i < InputSize(); ++i) {
      const auto input_nbytes = Input(i).nbytes();
      context_.template CopyBytes<Context, Context>(
          input_nbytes,
          (const uint8_t*)Input(i).raw_data(),
          coalesced->template mutable_data<uint8_t>() + coalesced_offset);

      // Note: this could cause Input(i) to free it's data if
      // Output(i) and Input(i) alias each other. This is safe on a
      // GPU (as the copy will happen-before the free), but it's
      // worth mentioning.

      Output(i)->ResizeLike(Input(i));
      Output(i)->ShareExternalPointer(
          static_cast<void*>(
              coalesced->template mutable_data<uint8_t>() + coalesced_offset),
          Input(i).meta(),
          input_nbytes);
      coalesced_offset += roundToAlignment(input_nbytes);
    }
    return true;
  }
};

template <typename T, class Context>
class AccumulateHistogramOp : public Operator<Context> {
 public:
  AccumulateHistogramOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        lower_bound_(
            OperatorBase::GetSingleArgument<float>("lower_bound", 0.0)),
        upper_bound_(
            OperatorBase::GetSingleArgument<float>("upper_bound", 1.0)),
        num_buckets_(OperatorBase::GetSingleArgument<int>("num_buckets", 1)) {
    CAFFE_ENFORCE_GT(num_buckets_, 0);
    // 2 more for histograms < lower_bound, >= upper_bound respectively
    num_output_buckets_ = num_buckets_ + 2;
    accumulate_hist_ = std::vector<int64_t>(num_output_buckets_, 0);
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(X_IN);
    auto* X_data = X.template data<T>();
    int N = X.size();
    auto* cur_hist = Output(CUR_HIST);
    auto* acc_hist = Output(ACC_HIST);
    cur_hist->Resize(num_output_buckets_);
    acc_hist->Resize(num_output_buckets_);
    auto* cur_hist_data = cur_hist->template mutable_data<int64_t>();
    auto* acc_hist_data = acc_hist->template mutable_data<int64_t>();
    auto segment = (upper_bound_ - lower_bound_) / num_buckets_;
    math::Set<int64_t, Context>(
        num_output_buckets_, 0, cur_hist_data, &context_);

    for (int i = 0; i < N; i++) {
      int bucket_index = -1;
      if (X_data[i] < lower_bound_) {
        bucket_index = 0;
      } else if (X_data[i] >= upper_bound_) {
        bucket_index = num_buckets_ + 1;
      } else {
        bucket_index = (int)((X_data[i] - lower_bound_) / segment) + 1;
      }
      cur_hist_data[bucket_index] += 1;
      accumulate_hist_[bucket_index] += 1;
    }

    for (int i = 0; i < num_output_buckets_; i++) {
      acc_hist_data[i] = accumulate_hist_[i];
    }

    return true;
  }

 private:
  float lower_bound_;
  float upper_bound_;
  int num_buckets_;
  int num_output_buckets_;
  std::vector<int64_t> accumulate_hist_;

  INPUT_TAGS(X_IN);
  OUTPUT_TAGS(CUR_HIST, ACC_HIST);
};

template <class Context>
class RangeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(RangeOp)

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t, float, double>>::call(
        this, Input(0));
  }

  template <typename T>
  T readScalarInput(const int index) {
    if (std::is_same<Context, TensorCPU>::value) {
      return Input(index).template data<T>()[0];
    } else {
      local_.template CopyFrom<Context>(Input(index));
      return local_.template data<T>()[0];
    }
  }

  template <typename T>
  bool DoRunWithType() {
    T stop = 0;
    T start = 0;
    T step = 1;

    for (int i = 0; i < InputSize(); ++i) {
      CAFFE_ENFORCE_EQ(Input(0).ndim(), 0, "All inputs must be scalar.");
    }

    switch (InputSize()) {
      case 1:
        stop = readScalarInput<T>(0);
        break;
      case 2:
        start = readScalarInput<T>(0);
        stop = readScalarInput<T>(1);
        break;
      case 3:
        step = readScalarInput<T>(2);
        start = readScalarInput<T>(0);
        stop = readScalarInput<T>(1);
        break;
    }
    CAFFE_ENFORCE_NE(step, 0, "Step size cannot be 0.");
    int length;
    auto diff = stop - start;
    if (std::is_integral<T>::value) {
      // Avoid casting to and from floats in case it introduces rounding and
      // avoid mod because the compiler doesn't strip unused code until later.
      length = diff / step;
      if (length * step < diff) {
        length += 1;
      }
    } else {
      length = static_cast<int>(ceil(diff / step));
    }
    auto* output = Output(0);
    // Match numpy's behavior here.
    if (length <= 0) {
      output->Resize(0);
      // Called for the side effect of setting the data.
      output->template mutable_data<T>();
      return true;
    } else {
      output->Resize(length);
      return DoRunOnDevice<T>(start, step, output);
    }
  }

  template <typename T>
  bool DoRunOnDevice(const T& start, const T& step, Tensor<Context>* output);

 private:
  // local CPU tensor for copying constants.
  TensorCPU local_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILITY_OPS_H_

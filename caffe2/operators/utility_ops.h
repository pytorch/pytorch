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
// produced by some of the earlier operators before it is used.
class FreeOp : public OperatorBase {
 public:
  USE_SIMPLE_BASE_CTOR_DTOR(FreeOp);

  bool Run() {
    for (Blob* output : Outputs()) {
      output->Reset();
    }
    return true;
  }

  INPUT_OUTPUT_STATS(0, INT_MAX, 1, INT_MAX);
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

  bool RunOnDevice() {
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
        device_context_.template Copy<T, Context, CPUContext>(
            input.size(), input.template data<T>(),
            temp_tensor.template mutable_data<T>());
      }
      std::stringstream values_stream;
      int total_count = std::min(temp_tensor.size(), limit_);
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
  INPUT_OUTPUT_STATS(1, INT_MAX, 0, 0);
  DISABLE_COPY_AND_ASSIGN(PrintOp);
};


template <class Context>
class AliasOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AliasOp);

  bool RunOnDevice() {
    auto& input = Input(0);
    CAFFE_DCHECK_GT(input.size(), 0);
    if (Output(0) == &input) {
      // If one calls an AliasOp but in fact it is in-place (input and output
      // are the same tensor), we will simply skip.
      return true;
    } else {
      Output(0)->ReshapeLike(input);
      Output(0)->ShareData(input);
    }
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(AliasOp);
};

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FlattenOp);

  bool RunOnDevice() {
    auto& input = Input(0);
    CAFFE_DCHECK_GT(input.size(), 0);
    Output(0)->Reshape(
        std::vector<int>{input.dim(0), input.size() / input.dim(0)});
    Output(0)->ShareData(input);
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FlattenOp);
};

// Output shares the data of input(0), but reshapes it like input(1).
template <class Context>
class ReshapeLikeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ReshapeLikeOp);

  bool RunOnDevice() {
    auto* output = Output(0);
    CAFFE_DCHECK_EQ(Input(0).size(), Input(1).size());
    output->ReshapeLike(Input(1));
    output->ShareData(Input(0));
    return true;
  }

  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ReshapeLikeOp);
};

template <class Context>
class SplitOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SplitOp);

  bool RunOnDevice() {
    const auto& input = Input(0);
    for (int i = 0; i < OutputSize(); ++i) {
      auto* output = Output(i);
      output->ReshapeLike(input);
      output->ShareData(input);
    }
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, INT_MAX);
  DISABLE_COPY_AND_ASSIGN(SplitOp);
};

template <typename T, class Context>
class SumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumOp);

  bool RunOnDevice() {
    auto& input0 = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input0);
    T* output_data = output->template mutable_data<T>();
    if (InputSize() == 1) {
      device_context_.template Copy<T, Context, Context>(
          input0.size(), input0.template data<T>(),
          output_data);
    } else {
      // Add the first two
      math::Add(output->size(), input0.template data<T>(),
                Input(1).template data<T>(),
                output_data, &device_context_);
      // Add remaining.
      for (int i = 2; i < InputSize(); ++i) {
        math::Add(output->size(), output_data, Input(i).template data<T>(),
                  output_data, &device_context_);
      }
    }
    return true;
  }

  INPUT_OUTPUT_STATS(1, INT_MAX, 1, 1);
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

  bool RunOnDevice() {
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
        &device_context_);
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
          &device_context_);
    }
    return true;
  }

  INPUT_OUTPUT_STATS(2, INT_MAX, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(WeightedSumOp);
};

template <class Context, class DstContext, class SrcContext>
class CopyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(CopyOp);

  bool RunOnDevice() {
    auto& input = OperatorBase::Input<Tensor<SrcContext> >(0);
    auto* output = OperatorBase::Output<Tensor<DstContext> >(0);
    output->ReshapeLike(input);
    this->device_context_.template Memcpy<SrcContext, DstContext>(
      input.nbytes(),
      input.raw_data(),
      output->raw_mutable_data(input.meta()));
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
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

  bool RunOnDevice() {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<vector<int, Context> >(0);
    *output = input.shape();
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(RecordShapeOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_UTILITY_OPS_H_

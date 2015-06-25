#ifndef CAFFE2_OPERATORS_UTILITY_OPS_H_
#define CAFFE2_OPERATORS_UTILITY_OPS_H_

#include <fstream>
#include <sstream>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

const char kPrintFileExtension[] = ".log";

// FreeOp frees the content of the output blob. We allow it to take in input
// blobs purely for the reason that it can "wait" on the input blobs to be
// produced by some of the earlier operators before it is used.
class FreeOp : public OperatorBase {
 public:
  USE_SIMPLE_BASE_CTOR_DTOR(FreeOp);

  bool Run() final {
    for (Blob* output : Outputs()) {
      output->Reset();
    }
    return true;
  }

  INPUT_OUTPUT_STATS(0, INT_MAX, 1, INT_MAX);
  DISABLE_COPY_AND_ASSIGN(FreeOp);
};

template <typename dtype, class DeviceContext>
class PrintOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  PrintOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        to_file_(OperatorBase::GetSingleArgument<int>("to_file", 0)),
        limit_(OperatorBase::GetSingleArgument<int>("limit", 0)) {
    if (limit_ == 0) {
      limit_ = INT_MAX;
    }
    if (to_file_) {
      // We will output to file instead of printing on screen.
      const string& target_folder = ws->RootFolder();
      // We will write each individual tensor to its individual file.
      log_files_.resize(def().inputs_size());
      for (int i = 0; i < def().inputs_size(); ++i) {
        log_files_[i].reset(new std::ofstream(
            target_folder + "/" + def().inputs(i) + kPrintFileExtension,
            std::ofstream::out | std::ofstream::trunc));
        CHECK(log_files_[i]->good())
            << "Failed to open PrintOp file for tensor " << def().inputs(i)
            << ". rdstate() = " << log_files_[i]->rdstate();
      }
    }
  }

  ~PrintOp() {
    for (auto& log_file : log_files_) {
      log_file->close();
    }
  }

  bool RunOnDevice() final {
    Tensor<dtype, CPUContext> temp_tensor;
    for (int input_id = 0; input_id < InputSize(); ++input_id) {
      auto& input = Input(input_id);
      DCHECK_GT(input.size(), 0);
      temp_tensor.ReshapeLike(input);
      device_context_.template Copy<dtype, CPUContext, DeviceContext>(
          temp_tensor.mutable_data(), input.data(), input.size());
      std::stringstream dims_stream;
      for (const int dim : input.dims()) {
        dims_stream << dim << ",";
      }
      std::stringstream values_stream;
      int total_count = std::min(temp_tensor.size(), limit_);
      const dtype* temp_tensor_data = temp_tensor.data();
      for (int i = 0; i < total_count - 1; ++i) {
        values_stream << temp_tensor_data[i] << ",";
      }
      // We do not add a comma after the last item.
      values_stream << temp_tensor_data[total_count - 1];
      if (to_file_) {
        // Also log to file.
        auto& stream = *log_files_[input_id];
        stream << values_stream.str();
        stream << std::endl;
      } else {
        // Log to console.
        LOG(INFO) << "Tensor " << def().inputs(input_id)
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

template <typename dtype, class DeviceContext>
class AliasOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AliasOp);

  bool RunOnDevice() final {
    auto& input = Input(0);
    DCHECK_GT(input.size(), 0);
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
  DISABLE_COPY_AND_ASSIGN(AliasOp);
};

template <typename dtype, class DeviceContext>
class FlattenOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FlattenOp);

  bool RunOnDevice() final {
    auto& input = Input(0);
    DCHECK_GT(input.size(), 0);
    Output(0)->Reshape(
        std::vector<int>{input.dim(0), input.size() / input.dim(0)});
    Output(0)->ShareData(input);
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FlattenOp);
};

// Output shares the data of input(0), but reshapes it like input(1).
template <typename dtype, class DeviceContext>
class ReshapeLikeOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ReshapeLikeOp);

  bool RunOnDevice() final {
    auto* output = Output(0);
    DCHECK_EQ(Input(0).size(), Input(1).size());
    output->ReshapeLike(Input(1));
    output->ShareData(Input(0));
    return true;
  }

  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ReshapeLikeOp);
};

template <typename dtype, class DeviceContext>
class SplitOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SplitOp);

  bool RunOnDevice() final {
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

template <typename dtype, class DeviceContext>
class SumOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumOp);

  bool RunOnDevice() final {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input);
    device_context_.template Copy<dtype, DeviceContext, DeviceContext>(
        output->mutable_data(), input.data(), input.size());
    for (int i = 1; i < InputSize(); ++i) {
      math::Add(output->size(), output->data(), Input(i).data(),
                output->mutable_data(), &device_context_);
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
template <typename dtype, class DeviceContext>
class WeightedSumOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(WeightedSumOp);

  bool RunOnDevice() final {
    DCHECK_EQ(InputSize() % 2, 0);
    auto& X0 = Input(0);
    auto& weight0 = Input(1);
    DCHECK_GT(X0.size(), 0);
    DCHECK_EQ(weight0.size(), 1);
    int size = X0.size();
    auto* output = Output(0);
    output->ReshapeLike(X0);
    math::Scale<dtype, DeviceContext>(
        size, weight0.data(), X0.data(), output->mutable_data(),
        &device_context_);
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
      math::Axpy<dtype, DeviceContext>(
          size, weight.data(), X.data(), output->mutable_data(),
          &device_context_);
    }
    return true;
  }

  INPUT_OUTPUT_STATS(2, INT_MAX, 1, 1);
  DISABLE_COPY_AND_ASSIGN(WeightedSumOp);
};

template <typename dtype, class DeviceContext,
          class DstContext, class SrcContext>
class CopyOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(CopyOp);

  bool RunOnDevice() final {
    auto& input = OperatorBase::Input<Tensor<dtype, SrcContext> >(0);
    auto* output = OperatorBase::Output<Tensor<dtype, DstContext> >(0);
    output->ReshapeLike(input);
    this->device_context_.template Copy<dtype, DstContext, SrcContext>(
      output->mutable_data(), input.data(), input.size());
    return true;
  }

  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CopyOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_UTILITY_OPS_H_

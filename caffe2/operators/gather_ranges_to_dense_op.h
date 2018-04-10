#ifndef CAFFE2_OPERATORS_GATHER_RANGES_TO_DENSE_OPS_H_
#define CAFFE2_OPERATORS_GATHER_RANGES_TO_DENSE_OPS_H_

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
class GatherRangesToDenseOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GatherRangesToDenseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        lengths_(OperatorBase::GetRepeatedArgument<int>("lengths")) {
    CAFFE_ENFORCE_GT(lengths_.size(), 0, "There has to be at least one length");
    for (auto length : lengths_) {
      CAFFE_ENFORCE_GT(length, 0, "Each length should be positive");
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(RANGES));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& ranges = Input(RANGES);
    CAFFE_ENFORCE_EQ(data.ndim(), 1, "Data has to be 1-D");
    CAFFE_ENFORCE_EQ(ranges.ndim(), 3, "Ranges has to be 3-D");
    if (InputSize() == 3) {
      auto& key = Input(KEY);
      CAFFE_ENFORCE_EQ(key.ndim(), 1, "Key has to be 1-D");
      CAFFE_ENFORCE(
          key.meta().template Match<int64_t>(), "Key has to be type int64_t");
    }
    CAFFE_ENFORCE_EQ(
        ranges.dim(1),
        lengths_.size(),
        "Nummber of ranges should match number of lengths");
    CAFFE_ENFORCE_EQ(
        ranges.dim(1),
        OutputSize(),
        "Nummber of ranges should match number of outputs");
    CAFFE_ENFORCE_EQ(
        ranges.dim(2), 2, "Ranges last dimension should be of size 2");

    auto* rawData = static_cast<const char*>(data.raw_data());
    auto* rangesData = ranges.template data<Index>();
    int rangesDataOffset = 0;
    auto itemsize = data.meta().itemsize();

    auto batchSize = ranges.dim(0);
    vector<TIndex> outputDims{batchSize, 0};
    vector<char*> outputRawData;
    for (int i = 0; i < OutputSize(); ++i) {
      auto* output = Output(i);
      outputDims[1] = lengths_[i];
      output->Resize(outputDims);
      char* ptr = static_cast<char*>(output->raw_mutable_data(data.meta()));
      memset(ptr, 0, output->nbytes());
      outputRawData.push_back(ptr);
    }

    for (int i = 0; i < batchSize; ++i) {
      for (int j = 0; j < OutputSize(); ++j) {
        auto rangeStart = rangesData[rangesDataOffset++];
        auto rangeLength = rangesData[rangesDataOffset++];
        if (rangeLength == 0) {
          // empty range, will be filled with zeros
          continue;
        }
        CAFFE_ENFORCE_EQ(
            rangeLength,
            lengths_[j],
            "Range lengths missmatch for output #",
            j);

        if (InputSize() == 2) {
          context_.template CopyItems<Context, Context>(
              data.meta(),
              rangeLength,
              rawData + rangeStart * itemsize,
              outputRawData[j] + i * itemsize * lengths_[j]);
        } else {
          auto& key = Input(KEY);
          auto* key_data = key.template data<int64_t>();
          vector<std::pair<int64_t, const char*>> buffer;
          for (int b_i = 0; b_i < rangeLength; ++b_i) {
            int64_t one_key_item = key_data[rangeStart + b_i];
            auto* one_data_item = rawData + (rangeStart + b_i) * itemsize;
            buffer.emplace_back(one_key_item, one_data_item);
          }
          std::sort(
              buffer.begin(),
              buffer.end(),
              [](const std::pair<int64_t, const char*>& left,
                 const std::pair<int64_t, const char*>& right) {
                return left.first < right.first;
              });
          for (int b_i = 0; b_i < rangeLength; ++b_i) {
            // Since this CPU only, directly copy to the destination.
            std::memcpy(
                outputRawData[j] + (i * lengths_[j] + b_i) * itemsize,
                buffer[b_i].second,
                itemsize);
          }
        }
      }
    }
    CAFFE_ENFORCE_EQ(rangesDataOffset, ranges.size());

    return true;
  }

  INPUT_TAGS(DATA, RANGES, KEY);

 private:
  vector<int> lengths_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GATHER_RANGES_TO_DENSE_OPS_H_

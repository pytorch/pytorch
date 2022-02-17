#ifndef CAFFE2_OPERATORS_GATHER_RANGES_TO_DENSE_OPS_H_
#define CAFFE2_OPERATORS_GATHER_RANGES_TO_DENSE_OPS_H_

#include <math.h>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include <c10/util/irange.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

#include <cstring>
#include <map>
#include <utility>

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(GatherRangesToDense);

namespace caffe2 {
template <class Context>
class GatherRangesToDenseOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit GatherRangesToDenseOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        lengths_(this->template GetRepeatedArgument<int>("lengths")),
        minObservation_(this->template GetSingleArgument<int64_t>(
            "min_observation",
            10000)),
        maxMismatchedRatio_(this->template GetSingleArgument<float>(
            "max_mismatched_ratio",
            0.01)),
        maxEmptyRatio_(
            this->template GetSingleArgument<float>("max_empty_ratio", 1.0)) {
    CAFFE_ENFORCE_GT(lengths_.size(), 0, "There has to be at least one length");
    for (auto length : lengths_) {
      CAFFE_ENFORCE_GT(length, 0, "Each length should be positive");
    }
    CAFFE_ENFORCE_GT(
        minObservation_, 0, "The number of observations is at least 1");
    // Initialize the empty and mismatch counter.
    for (const auto i : c10::irange(OutputSize())) {
      (void)i; // Suppress unused variable warning
      emptyRanges_.push_back(0);
      mismatchedRanges_.push_back(0);
      mismatchedLengths_.push_back(set<int>());
    }
  }

  ~GatherRangesToDenseOp() noexcept override {
    if (totalRanges_ > minObservation_) {
      string debugString;
      if (this->has_debug_def()) {
        debugString =
            "Info from operator: " + ProtoDebugString(this->debug_def());
      } else {
        debugString = "Info from operator: no op def";
      }

      LOG(INFO) << "In GatherRangesToDenseOp:\n"
                << "  Lifetime empty ranges for each feature is "
                << emptyRanges_ << ".\n"
                << "  Lifetime mismatched ranges for each feature is "
                << mismatchedRanges_ << ".\n"
                << "  With a total of " << totalRanges_ << " examples.\n"
                << debugString;
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(RANGES, CPU));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& ranges = Input(RANGES);
    CAFFE_ENFORCE_EQ(data.dim(), 1, "Data has to be 1-D");
    CAFFE_ENFORCE_EQ(ranges.dim(), 3, "Ranges has to be 3-D");
    if (InputSize() == 3) {
      auto& key = Input(KEY);
      CAFFE_ENFORCE_EQ(key.dim(), 1, "Key has to be 1-D");
      CAFFE_ENFORCE(
          key.dtype().template Match<int64_t>(), "Key has to be type int64_t");
    }
    CAFFE_ENFORCE_EQ(
        ranges.size(1),
        lengths_.size(),
        "Number of ranges should match number of lengths");
    CAFFE_ENFORCE_EQ(
        ranges.size(1),
        OutputSize(),
        "Number of ranges should match number of outputs");
    CAFFE_ENFORCE_EQ(
        ranges.size(2), 2, "Ranges last dimension should be of size 2");

    auto* rawData = static_cast<const char*>(data.raw_data());
    auto* rangesData = ranges.template data<Index>();
    int rangesDataOffset = 0;
    auto itemsize = data.dtype().itemsize();

    const auto batchSize = ranges.size(0);
    vector<int64_t> outputDims{batchSize, 0};
    vector<char*> outputRawData;
    outputRawData.reserve(OutputSize());
    for (const auto i : c10::irange(OutputSize())) {
      auto *const output = Output(i);
      outputDims[1] = lengths_[i];
      output->Resize(outputDims);
      char *const ptr = static_cast<char*>(output->raw_mutable_data(data.dtype()));
      memset(ptr, 0, output->nbytes());
      outputRawData.push_back(ptr);
    }

    for (const auto i : c10::irange(batchSize)) {
      for (const auto j : c10::irange(OutputSize())) {
        const auto rangeStart = rangesData[rangesDataOffset++];
        const auto rangeLength = rangesData[rangesDataOffset++];

        if (rangeLength == 0) {
          // empty range, will be filled with zeros
          emptyRanges_[j]++;
          continue;
        }
        if (rangeLength != lengths_[j]) {
          // Range lengths missmatch for output #, will be filled with zeros
          // Note, empty ranges are not counted as mismatched because empty
          // are more common and more tolerable.
          mismatchedRanges_[j]++;
          mismatchedLengths_[j].insert(rangeLength);
          continue;
        }

        if (InputSize() == 2) {
          context_.CopyItemsSameDevice(
              data.dtype(),
              rangeLength,
              rawData + rangeStart * itemsize,
              outputRawData[j] + i * itemsize * lengths_[j]);
        } else {
          auto& key = Input(KEY);
          auto* key_data = key.template data<int64_t>();
          vector<std::pair<int64_t, const char*>> buffer;
          for (const auto b_i : c10::irange(rangeLength)) {
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
          for (const auto b_i : c10::irange(rangeLength)) {
            // Since this CPU only, directly copy to the destination.
            std::memcpy(
                outputRawData[j] + (i * lengths_[j] + b_i) * itemsize,
                buffer[b_i].second,
                itemsize);
          }
        }
      }
    }

    CAFFE_ENFORCE_EQ(rangesDataOffset, ranges.numel());

    // Check whether the empty and mismatch ratio exceeded the threshold.
    totalRanges_ += batchSize;
    for (const auto j : c10::irange(OutputSize())) {
      // Only check when the ratio is not set to allow all mismatches.
      if (maxMismatchedRatio_ < 1.0) {
        CAFFE_ENFORCE_GE(
            std::max(totalRanges_, minObservation_) * maxMismatchedRatio_,
            mismatchedRanges_[j],
            "Ratio of range length mismatch for feature at index ",
            j,
            " is ",
            (static_cast<double>(mismatchedRanges_[j]) /
             static_cast<double>(totalRanges_)),
            " (",
            mismatchedRanges_[j],
            "/",
            totalRanges_,
            ") which exceeds ",
            maxMismatchedRatio_,
            ". The incorrect lengths include: ",
            mismatchedLengths_[j]);
      }

      // Only check when the ratio is not set to allow all examples to be empty.
      if (maxEmptyRatio_ < 1.0) {
        CAFFE_ENFORCE_GE(
            std::max(totalRanges_, minObservation_) * maxEmptyRatio_,
            emptyRanges_[j],
            "Ratio of empty ranges for feature at index ",
            j,
            " is ",
            (static_cast<double>(emptyRanges_[j]) /
             static_cast<double>(totalRanges_)),
            " (",
            emptyRanges_[j],
            "/",
            totalRanges_,
            ") which exceeds ",
            maxEmptyRatio_);
      }
    }

    return true;
  }

  INPUT_TAGS(DATA, RANGES, KEY);

 private:
  vector<int> lengths_;
  int64_t totalRanges_ = 0;
  vector<int64_t> emptyRanges_;
  vector<int64_t> mismatchedRanges_;
  vector<set<int>> mismatchedLengths_;
  // To avoid false alarm due to insufficient sample (e.g., first batch being
  // mismatched and causing 100% to be mismatched), use a threshold to ensure
  // enough samples are gathered before decideding whether there is an alarm or
  // not.
  int64_t minObservation_ = 0;
  float maxMismatchedRatio_ = 0;
  float maxEmptyRatio_ = 0;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GATHER_RANGES_TO_DENSE_OPS_H_

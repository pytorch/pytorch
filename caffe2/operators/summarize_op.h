#ifndef CAFFE2_OPERATORS_SUMMARIZE_OP_H_
#define CAFFE2_OPERATORS_SUMMARIZE_OP_H_

#include <fstream>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

constexpr char kSummaryzeOpExtension[] = ".summary";

// Accumulate operator accumulates the input tensor to the output tensor. If the
// output tensor already has the right size, we add to it; otherwise, we first
// initialize the output tensor to all zeros, and then do accumulation. Any
// further calls to the operator, given that no one else fiddles with the output
// in the interim, will do simple accumulations.
template <typename T, class Context>
class SummarizeOp final : public Operator<Context> {
 public:
  SummarizeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        to_file_(OperatorBase::GetSingleArgument<int>("to_file", 0)) {
    if (to_file_) {
      // We will output to file instead of printing on screen.
      const string& target_folder = ws->RootFolder();
      // We will write each individual tensor to its individual file.
      log_file_.reset(new std::ofstream(
          target_folder + "/" + def.input(0) + kSummaryzeOpExtension,
          std::ofstream::out | std::ofstream::trunc));
      CAFFE_CHECK(log_file_->good())
          << "Failed to open summarize file for tensor " << def.input(0)
          << ". rdstate() = " << log_file_->rdstate();
    }
  }
  ~SummarizeOp() { if (to_file_) log_file_->close(); }
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

  static constexpr int MIN_IDX = 0;
  static constexpr int MAX_IDX = 1;
  static constexpr int MEAN_IDX = 2;
  static constexpr int STD_IDX = 3;
  static constexpr int NUM_STATS = 4;

 protected:
  bool to_file_;
  std::unique_ptr<std::ofstream> log_file_;
  // Input: X; output: if set, a summarized vector of shape 4, with the values
  // being min, max, mean and std respectively.
  INPUT_OUTPUT_STATS(1, 1, 0, 1);
  DISABLE_COPY_AND_ASSIGN(SummarizeOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_SUMMARIZE_OP_H_

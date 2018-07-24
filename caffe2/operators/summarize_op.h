#ifndef CAFFE2_OPERATORS_SUMMARIZE_OP_H_
#define CAFFE2_OPERATORS_SUMMARIZE_OP_H_

#include <fstream>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

constexpr char kSummaryzeOpExtension[] = ".summary";

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
      // Also, since the namescope is currently represented by "/", we will
      // need to replace it with a symbol that does not conflict with the
      // folder separator in Linux.
      string proper_name = def.input(0);
      std::replace(proper_name.begin(), proper_name.end(), '/', '#');
      log_file_.reset(new std::ofstream(
          target_folder + "/" + proper_name + kSummaryzeOpExtension,
          std::ofstream::out | std::ofstream::trunc));
      CAFFE_ENFORCE(
          log_file_->good(),
          "Failed to open summarize file for tensor ",
          def.input(0),
          ". rdstate() = ",
          log_file_->rdstate());
    }
  }
  ~SummarizeOp() {
    if (to_file_)
      log_file_->close();
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

  static constexpr const int& MIN_IDX = 0;
  static constexpr const int& MAX_IDX = 1;
  static constexpr const int& MEAN_IDX = 2;
  static constexpr const int& STD_IDX = 3;

  static constexpr const int& NUM_STATS = 4;

 protected:
  bool to_file_;
  std::unique_ptr<std::ofstream> log_file_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SUMMARIZE_OP_H_

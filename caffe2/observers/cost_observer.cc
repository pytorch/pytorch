#include "cost_observer.h"
#include "caffe2/core/common.h"

namespace caffe2 {

CostObserver::CostObserver(NetBase* subject_)
    : ObserverBase<NetBase>(subject_),
      detailedOpStats_(subject_->GetOperators().size()),
      net_name_(subject_->Name()) {
  const auto& ops = subject_->GetOperators();
  for (int i = 0; i < ops.size(); i++) {
    ops[i]->AttachObserver(
        caffe2::make_unique<CostOpObserver>(ops[i], &detailedOpStats_[i]));
  }
}

CostObserver::~CostObserver() {
  // put logging under the lock so that logs from different nets don't overlap
  // perf is not of concern since it's debug mode and we're talking only about
  // destruction
  static std::mutex loggingMutex;
  std::lock_guard<std::mutex> lock(loggingMutex);

  CaffeMap<string, OpSchema::Cost> cost_per_op_type = getCostPerOpType();
  // sort by decreasing flops.
  std::vector<std::pair<std::string, OpSchema::Cost>> cost_per_op_type_vec(
      cost_per_op_type.begin(), cost_per_op_type.end());
  std::sort(
      cost_per_op_type_vec.begin(),
      cost_per_op_type_vec.end(),
      [](const std::pair<std::string, OpSchema::Cost>& left,
         const std::pair<std::string, OpSchema::Cost>& right) {
        return left.second.flops > right.second.flops;
      });
  LOG(INFO) << "================ Detailed stats for net " << net_name_
            << " ================";
  LOG(INFO) << "Cost (flops, bytes_read, bytes_written) per operator type:";
  for (const auto& item : cost_per_op_type_vec) {
    LOG(INFO) << std::setw(15) << std::setfill(' ') << item.second.flops << " "
              << item.second.bytes_read << " " << item.second.bytes_written
              << " " << item.first;
  }
}

CostOpObserver::CostOpObserver(OperatorBase* op, DetailedStat* stat)
    : ObserverBase<OperatorBase>(op), stat_(stat) {
  stat->opType = op->debug_def().type();
  stat->displayName =
      (op->debug_def().name().size()
           ? op->debug_def().name()
           : (op->debug_def().output_size() ? op->debug_def().output(0)
                                            : "NO_OUTPUT"));
}

void CostObserver::Start() {}

void CostObserver::Stop() {}

void CostOpObserver::Start() {
  if (subject_->HasAsyncPart()) {
    LOG(INFO) << "Not printing shape info for an async operator";
  } else {
    if (subject_->debug_def().type() != "FC" && subject_->InputSize() > 1) {
      return;
    }
    if (subject_->debug_def().type() == "FC") {
      const vector<TensorShape>& in = subject_->InputTensorShapes();
      const OperatorDef& def = subject_->debug_def();
      ArgumentHelper helper(def);

      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const auto canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      const uint64_t M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));
      const uint64_t K = size_from_dim_(canonical_axis, GetDimsVector(in[0]));
      auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
      const int canonical_axis_w =
          canonical_axis_index_(axis_w, in[1].dims().size());
      const uint64_t N = size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));

      stat_->c.flops += M * N * (2 * K + 1);
      stat_->c.bytes_read += (K * (M + N) + N);
      stat_->c.bytes_written += M * N;
      stat_->c.params_bytes = (K * N + N);
    } else if (subject_->debug_def().type() == "Log") {
      const auto& blob = subject_->InputBlob(0);
      auto tensor_info_fun = GetTensorInfoFunction(blob.meta().id());
      if (tensor_info_fun) {
        size_t capacity;
        DeviceOption device;
        vector<int64_t> shape =
            tensor_info_fun(blob.GetRaw(), &capacity, &device);
        uint64_t input_size = 1;
        for (int i = 0; i < shape.size(); i++) {
          input_size *= shape[i];
        }
        stat_->c.flops += input_size;
        stat_->c.bytes_read += input_size;
        stat_->c.bytes_written += input_size;
      }
    }
  }
}

void CostOpObserver::Stop() {}

} // namespace caffe2

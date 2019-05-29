#pragma once

#include <sstream>

#include "caffe2/intra_op_parallel/intra_op_parallel.h"

namespace caffe2 {

namespace intra_op_parallel {

class ParallelBatchMatMulOp final : public ParallelOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  ParallelBatchMatMulOp(const OperatorDef& operator_def, Workspace* ws);
  virtual ~ParallelBatchMatMulOp() override;

 private:
  bool RunOnDevicePrologue(int num_tasks) override;
  bool RunOnDeviceParallel(int task_id, int num_tasks) override;

 protected:
  bool trans_a_;
  bool trans_b_;
  bool broadcast_;

  size_t M_, N_, K_;
  size_t A_stride_, B_stride_, Y_stride_;
  size_t num_sub_batches_, num_outer_batches_;
};

} // namespace intra_op_parallel

} // namespace caffe2

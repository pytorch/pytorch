#ifndef CAFFE2_NET_JIT_H
#define CAFFE2_NET_JIT_H

#include <vector>

#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

enum JITOpCode {
  NONE = 0,
  C2_OP = 1, // run C2 operator
  FORK = 2, // launch an async task
  JOIN = 3, // wait for async task(s)
  RETURN = 4, // finish task execution
  // No IF, GOTO since there's no branching or loops in C2 graphs
};

class JITOp {
 public:
  // not using polymorphism for perf. reasons
  static JITOp ForkOp(
      int future_id,
      size_t task_address,
      const std::vector<int>& input_future_ids) {
    JITOp op(JITOpCode::FORK);
    op.arg1_ = future_id;
    op.arg2_ = task_address;
    op.argv_ = input_future_ids;
    return op;
  }

  static JITOp JoinOp(const std::vector<int>& future_ids) {
    JITOp op(JITOpCode::JOIN);
    op.argv_ = future_ids;
    return op;
  }

  static JITOp C2Op(size_t op_id) {
    JITOp op(JITOpCode::C2_OP);
    op.arg1_ = op_id;
    return op;
  }

  static JITOp ReturnOp() {
    return JITOp(JITOpCode::RETURN);
  }

  JITOpCode GetOpCode() const {
    return op_code_;
  }

  // Fork
  size_t GetTaskAddress() const {
    CAFFE_ENFORCE_EQ(op_code_, JITOpCode::FORK, "Expected FORK op");
    return (size_t)arg2_;
  }

  int GetFutureId() const {
    CAFFE_ENFORCE_EQ(op_code_, JITOpCode::FORK, "Expected FORK op");
    return arg1_;
  }

  const std::vector<int>& GetInputFutureIds() const {
    CAFFE_ENFORCE_EQ(op_code_, JITOpCode::FORK, "Expected FORK op");
    return argv_;
  }

  void SetTaskAddress(size_t task_address) {
    CAFFE_ENFORCE_EQ(op_code_, JITOpCode::FORK, "Expected FORK op");
    arg2_ = task_address;
  }

  // Join
  const std::vector<int>& GetFutureIds() const {
    CAFFE_ENFORCE_EQ(op_code_, JITOpCode::JOIN, "Expected JOIN op");
    return argv_;
  }

  // C2Op
  size_t GetOpId() const {
    CAFFE_ENFORCE_EQ(op_code_, JITOpCode::C2_OP, "Expected C2 op");
    return arg1_;
  }

  std::string DebugStr() const {
    switch (op_code_) {
      case C2_OP: {
        return "C2: op " + caffe2::to_string(GetOpId());
      }
      case FORK: {
        return "FORK: addr " + caffe2::to_string(GetTaskAddress()) +
            ", future " + caffe2::to_string(GetFutureId()) + ", depends on " +
            debugFuturesStr(GetInputFutureIds());
      }
      case JOIN: {
        return "JOIN: depends on " + debugFuturesStr(GetFutureIds());
      }
      case RETURN: {
        return "RETURN";
      }
      default: {
        CAFFE_THROW("Invalid opcode: ", op_code_);
      }
    }
  }

 protected:
  explicit JITOp(JITOpCode op_code) : op_code_(op_code) {}
  std::string debugFuturesStr(const std::vector<int>& future_ids) const {
    std::string debug_str = "[";
    for (auto fid : future_ids) {
      if (debug_str.empty()) {
        debug_str = caffe2::to_string(fid);
      } else {
        debug_str += ", " + caffe2::to_string(fid);
      }
    }
    debug_str += "]";
    return debug_str;
  }

 private:
  JITOpCode op_code_;
  int arg1_;
  int arg2_;
  std::vector<int> argv_;
};

// Generates JIT program from a C2 net.
//
// A JIT program is a sequence of instructions, each instruction has one
// of the following types:
//  - C2_OP: calls a corresponding C2 op, args:
//           op's index in c2_operators_;
//  - FORK: launches a parallel execution, args:
//           returned Future can be specified later using a unique id
//           passed in the first arg;
//           start address of an async task (instruction index);
//           unique ids of the Futures passed as inputs into the task
//  - JOIN: suspends execution of the current task until the given Futures
//          are completed, args:
//          vector of unique ids identifying the Futures;
//  - RETURN: finishes a task's execution,
//            must be always the last instruction of a task
class JITC2Program {
 public:
  JITC2Program(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  const std::vector<JITOp>& GetOps() const;
  const std::vector<OperatorBase*>& GetC2Ops() const;

 private:
  void emit(const JITOp& op);
  size_t nextAddress() const;
  size_t taskToId(size_t task_id) const;

  size_t numTasks() const;
  const std::vector<int>& parents(size_t task_id) const;
  const std::vector<int>& children(size_t task_id) const;
  const std::vector<int>& taskOps(size_t task_id) const;
  static std::vector<int> sequence(size_t length);

  void traverseTasks(std::function<void(int)> visitor);
  void initGraph(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  std::vector<JITOp> ops_;
  std::vector<OperatorBase*> c2_operators_;

  // legacy graph structs
  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<dag_utils::OpGraphNode> chain_nodes_;
  std::vector<std::vector<int>> chains_;
};

} // namespace caffe2

#endif // CAFFE2_NET_JIT_H

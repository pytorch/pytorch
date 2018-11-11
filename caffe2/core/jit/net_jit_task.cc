#include "caffe2/core/jit/net_jit_task.h"

#include <functional>

#include "caffe2/core/jit/net_jit_constants.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

JITC2Task::JITC2Task(
    const std::shared_ptr<JITC2Program>& program,
    const size_t start_address,
    JITC2TaskRunner& task_runner,
    bool use_dfs_scheduling,
    const std::vector<JITFuture*>& inputs,
    const std::unordered_map<int, JITFuture*>& local_vars)
    : program_(program),
      start_address_(start_address),
      address_(start_address),
      inputs_(inputs),
      task_runner_(task_runner),
      cpu_async_event_(nullptr),
      local_vars_(local_vars),
      use_dfs_scheduling_(use_dfs_scheduling) {
  CAFFE_ENFORCE_LT(
      inputs.size(),
      jit::MAX_FUTURE_INPUTS,
      "Too many future inputs: ",
      inputs.size());
  init();

  // registering JIT ops handlers
  _REGISTER_OP_HANDLER(C2_OP, handleC2Op);
  _REGISTER_OP_HANDLER(FORK, handleFork);
  _REGISTER_OP_HANDLER(JOIN, handleJoin);
  _REGISTER_OP_HANDLER(RETURN, handleReturn);
}

JITFuture* JITC2Task::Run() {
  try {
    const auto& ops = program_->GetOps();
    CAFFE_ENFORCE_LT(
        start_address_,
        ops.size(),
        "Invalid JIT task start address: ",
        start_address_);

    for (address_ = start_address_; address_ < ops.size(); ++address_) {
      if (!handleOp(ops.at(address_))) {
        break;
      }
    }
  } catch (const std::exception& e) {
    future_.SetCompleted(e.what());
  } catch (...) {
    future_.SetCompleted("Unknown error");
  }
  return &future_;
}

JITFuture& JITC2Task::GetFuture() {
  return future_;
}

const JITFuture& JITC2Task::GetFuture() const {
  return future_;
}

////

bool JITC2Task::handleC2Op(const JITOp& op) {
  auto op_id = op.GetOpId();
  const auto& ops = program_->GetC2Ops();
  CAFFE_ENFORCE(op_id >= 0 && op_id < ops.size(), "Invalid C2 op id: ", op_id);
  CAFFE_ENFORCE(
      ops[op_id]->RunAsync(0), // using logical stream_id = 0
      "Error in C2 op, id: ",
      op_id,
      ", type: ",
      ops[op_id]->type());
  return true;
}

std::vector<JITFuture*> JITC2Task::futures(const std::vector<int>& future_ids) {
  std::vector<JITFuture*> input_futures;
  if (!future_ids.empty()) {
    input_futures.reserve(future_ids.size());
    for (auto future_id : future_ids) {
      if (local_vars_.count(future_id)) {
        input_futures.push_back(local_vars_[future_id]);
      } else if (future_id >= 0 && future_id < inputs_.size()) {
        input_futures.push_back(inputs_[future_id]);
      } else {
        CAFFE_THROW("Undefined local future id: ", future_id);
      }
    }
  }
  return input_futures;
}

bool JITC2Task::handleFork(const JITOp& op) {
  auto future_id = op.GetFutureId();
  auto task_address = op.GetTaskAddress();
  const auto& input_futures = futures(op.GetInputFutureIds());
  auto fork_task = std::make_shared<JITC2Task>(
      program_, task_address, task_runner_, use_dfs_scheduling_, input_futures);
  if (use_dfs_scheduling_) {
    local_vars_[future_id] = fork_task->Run();
    // make sure task is alive as long as parent task is alive
    inline_tasks_.push_back(fork_task);
  } else {
    local_vars_[future_id] = task_runner_.RunTask(fork_task);
  }
  return true;
}

bool JITC2Task::handleJoin(const JITOp& op) {
  // executes the rest of the task
  auto cont_func = [this](const JITFuture* f, bool in_callback) {
    if (!f->IsFailed()) {
      // continue execution starting from the next JIT op
      auto cont_task = std::make_shared<JITC2Task>(
          program_,
          address_ + 1,
          task_runner_,
          use_dfs_scheduling_,
          inputs_,
          local_vars_);

      JITFuture* cont_future;
      if (use_dfs_scheduling_ && !in_callback) {
        cont_future = cont_task->Run();
        // make sure task is alive as long as parent task is alive
        inline_tasks_.push_back(cont_task);
      } else {
        cont_future = task_runner_.RunTask(cont_task);
      }

      cont_future->SetCallback([this](const JITFuture* cf) {
        if (!cf->IsFailed()) {
          future_.SetCompleted();
        } else {
          future_.SetCompleted(cf->ErrorMessage().c_str());
        }
      });
    } else {
      future_.SetCompleted(f->ErrorMessage().c_str());
    }
  };

  const auto& join_futures = futures(op.GetFutureIds());
  joined_future_ = caffe2::make_unique<JITFuture>(join_futures);

  // make sure we use DFS scheduling only when the future is ready,
  // otherwise use thread pool from within a callback
  if (joined_future_->IsCompleted()) {
    cont_func(joined_future_.get(), false);
  } else {
    joined_future_->SetCallback(
        [this, cont_func](const JITFuture* f) { cont_func(f, true); });
  }

  return false;
}

bool JITC2Task::handleReturn(const JITOp& op) {
  if (IsCPUDeviceType(device_option_.device_type())) {
    if (cpu_async_event_) {
      // this is a CPU async op, use it's Event to set callback
      cpu_async_event_->SetCallback([this]() {
        auto status = cpu_async_event_->Query();
        if (status == EventStatus::EVENT_SUCCESS) {
          future_.SetCompleted();
        } else {
          future_.SetCompleted(cpu_async_event_->ErrorMessage().c_str());
        }
      });
    } else {
      // this a sync CPU op
      future_.SetCompleted();
    }
  } else {
    // TODO: handle CUDA case here
    future_.SetCompleted("CUDA not supported");
  }
  return false;
}

bool JITC2Task::handleOp(const JITOp& op) {
  auto opcode = op.GetOpCode();
  CAFFE_ENFORCE(
      handler_registry_.count(opcode), "Unknown JIT opcode: ", opcode);
  return handler_registry_[opcode](op);
}

const DeviceOption& JITC2Task::GetDeviceOption() const {
  return device_option_;
}

void JITC2Task::init() {
  // in a C2 task all C2 ops have the same device option;
  // find the first DeviceOption, or use a default one
  const auto& ops = program_->GetOps();
  const auto& c2_ops = program_->GetC2Ops();
  bool found_device = false;
  for (auto address = start_address_; address < ops.size(); ++address) {
    const auto& op = ops.at(address);
    if (op.GetOpCode() == JITOpCode::C2_OP) {
      auto c2_op_id = op.GetOpId();
      CAFFE_ENFORCE(
          c2_op_id >= 0 && c2_op_id < c2_ops.size(), "Invalid JIT C2 op");
      device_option_ = c2_ops.at(c2_op_id)->device_option();
      found_device = true;

      // Check the case of an CPU async op and save its Event
      if (IsCPUDeviceType(device_option_.device_type()) &&
          c2_ops.at(c2_op_id)->HasAsyncPart()) {
        cpu_async_event_ = &c2_ops.at(c2_op_id)->event();
      }

      break;
    }
  }
  if (!found_device) {
    device_option_ = caffe2::DeviceOption(); // default device option
    device_option_.set_device_type(PROTO_CPU);
  }
}

} // namespace caffe2

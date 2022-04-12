#include "caffe2/core/net_async_task.h"

#include "caffe2/core/net_async_task_graph.h"

namespace caffe2 {

// NOLINTNEXTLINE(modernize-pass-by-value)
AsyncTask::AsyncTask(const std::vector<OperatorBase*>& ops) : ops_(ops) {
  CAFFE_ENFORCE(!ops_.empty());
  device_option_ = ops_.front()->device_option();
  for (auto& op : ops_) {
    CAFFE_ENFORCE(IsSameDevice(device_option_, op->device_option()));
  }
  Reset();
}

void AsyncTask::handleChainError(
    OperatorBase* op,
    const char* err_str,
    bool save_exception) {
  std::string err_msg = err_str;
  if (op) {
    err_msg += ",  op " + (op->has_debug_def() ? op->type() : " unknown");
  }
  LOG(ERROR) << err_msg;

  // save error message and exception in chain's Event
  auto last_op = ops_.back();
  if (save_exception) {
    last_op->event().SetFinishedWithException(err_msg.c_str());
  } else {
    last_op->event().SetFinished(err_msg.c_str());
  }

  // set future as completed with an error
  // TODO: exceptions in future
  future_.SetCompleted(err_msg.c_str());
}

bool AsyncTask::Run(const ExecutionOptions& options) {
  // TODO: insert CUDA's async stream waits; tracing and counters
  OperatorBase* op = nullptr;
  try {
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (auto op_idx = 0U; op_idx < ops_.size(); ++op_idx) {
      op = ops_[op_idx];
      int stream_id = 0; // TODO: thread local stream id
      if (!op->RunAsync(stream_id)) {
        handleChainError(op, "Failed to execute an op");
        return false;
      }
    }

    if (options.finish_chain_) {
      op = ops_.back();
      op->Finish();
    }

    // set the future as successfully completed or, in case of async CPU,
    // use op's callback
    if (IsCPUDeviceType(device_option_.device_type()) &&
        ops_.back()->HasAsyncPart()) {
      auto& event = ops_.back()->event();
      event.SetCallback([this, &event]() {
        CAFFE_ENFORCE(event.IsFinished());
        if (event.Query() == EventStatus::EVENT_SUCCESS) {
          future_.SetCompleted();
        } else {
          // TODO: support for exceptions
          future_.SetCompleted(event.ErrorMessage().c_str());
        }
      });
    } else {
      future_.SetCompleted();
    }
  } catch (const std::exception& e) {
    handleChainError(op, e.what(), /* save_exception */ true);
    return false;
  } catch (...) {
    handleChainError(
        op,
        "Failed to execute task: unknown error",
        /* save_exception */ true);
    return false;
  }

  return true;
}

void AsyncTask::Reset() {
  for (auto& op : ops_) {
    op->ResetEvent();
  }
  future_.ResetState();
}

DeviceOption AsyncTask::GetDeviceOption() const {
  return device_option_;
}

AsyncTaskFuture& AsyncTask::GetFuture() {
  return future_;
}

const AsyncTaskFuture& AsyncTask::GetFuture() const {
  return future_;
}

}; // namespace caffe2

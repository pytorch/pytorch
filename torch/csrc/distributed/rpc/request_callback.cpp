#include <torch/csrc/distributed/rpc/request_callback.h>

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

namespace {

// When request message has autograd info, processMessage() will set up valid
// current context id properly. This struct is used to clean up current context
// id after processMessage() is done.
struct ClearAutogradContextGuard {
  ClearAutogradContextGuard() = default;
  ~ClearAutogradContextGuard() {
    if (valid) {
      DistAutogradContainer::getInstance().clearCurrentContext();
    }
  }

  void cancel() {
    valid = false;
  }
  bool valid = true;
};

} // anonymous namespace

std::shared_ptr<FutureMessage> RequestCallback::operator()(
    Message& request) const {
  // For a recv thread, current context id should be invalid outside
  // processMessage().
  try {
    // do we need to run clear() in the future invocation?
    ClearAutogradContextGuard guard;
    auto ret = processMessage(request);
    if (!ret->completed()) {
      guard.cancel();
      // If not yet complete, clear the context when we do.
      ret->addCallback(
          [](const Message&) { ClearAutogradContextGuard guard2; });
    }
    return ret;
  } catch (std::exception& e) {
    LOG(ERROR) << "Received error while processing request type "
               << request.type() << ": " << e.what();
    return std::make_shared<FutureMessage>(createExceptionResponse(request, e));
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch

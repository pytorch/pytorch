#include <torch/csrc/distributed/rpc/request_callback.h>

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

namespace {

Message createException(const Message& request, const std::exception& e) {
  const char* err = e.what();
  std::vector<char> payload(err, err + strlen(err));
  return Message(
      std::move(payload),
      std::vector<torch::Tensor>(),
      MessageType::EXCEPTION,
      request.id());
}

// When request message has autograd info, processMessage() will set up valid
// current context id properly. This struct is used to clean up current context
// id after processMessage() is done.
struct ClearAutogradContextGuard {
  ClearAutogradContextGuard() = default;
  ~ClearAutogradContextGuard() {
    clear();
  }

  void clear() {
    auto& autogradContainer = DistAutogradContainer::getInstance();
    autogradContainer.clearCurrentContext();
  }
};

} // anonymous namespace

Message RequestCallback::operator()(Message& request) const {
  // For a recv thread, current context id should be invalid outside
  // processMessage().
  ClearAutogradContextGuard guard;
  try {
    return processMessage(request);
  } catch (std::exception& e) {
    LOG(ERROR) << "Received error while processing request type "
               << request.type() << ": " << e.what();
    return createException(request, e);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch

#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
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

} // anonymous namespace

Message RequestCallback::operator()(Message& request) {
  try {
    return processMessage(request);
  } catch (std::exception& e) {
    return createException(request, e);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch

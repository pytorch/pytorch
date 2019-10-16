#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

Message RequestCallback::operator()(Message& request) const {
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

#include <torch/csrc/distributed/rpc/request_callback.h>

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

c10::intrusive_ptr<JitFuture> RequestCallback::operator()(
    Message& request,
    std::shared_ptr<LazyStreamContext> ctx) const {
  // NB: cannot clear autograd context id here because the processMessage method
  // might pause waiting for all RRefs in the arguments to be confirmed by their
  // owners and resumne processing in a different thread. Hence, the
  // thread_local context id needs to be set and cleared in the thread that
  // indeed carries out the processing logic.
  return processMessage(request, std::move(ctx));
}

} // namespace rpc
} // namespace distributed
} // namespace torch

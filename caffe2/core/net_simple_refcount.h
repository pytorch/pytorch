#ifndef CAFFE2_CORE_NET_SIMPLE_REFCOUNT_H_
#define CAFFE2_CORE_NET_SIMPLE_REFCOUNT_H_

#include <vector>

#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_simple.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

// SimpleRefcountNet is an implementation that adds an additional abstraction
// on top of SimpleRefCountNet: it tracks all the tensors and for those that are
// considered internal/temporary, delete them once their refcount go to zero.
// In the context of a simple static run, this can be carried out during
// construction time: we will do a pass through the network and track what
// blobs we need to do reset on, after the execution of every op.
//
// To identify which blob is considered temporary, we employ the following
// strategy: any blob that is
// (1) consumed but not produced by ops in the net, or
// (2) produced but not consumed by ops in the net, or
// (3) is marked as external_output in the protobuf
// will NOT be considered temporary.
//
// In the long run, we should design proper functional interfaces so that
// nets are less imperative and more functional.
//
// Also, for now, SimpleRefCountNet should only be used for benchmarking
// purposes and not product use, since it is not going to provide better
// performance gain, and is implicitly incompatible with the contract that
// earlier Nets expose - that all intermediate blobs are visible to the users.
class SimpleRefCountNet final : public SimpleNet {
 public:
  SimpleRefCountNet(
      const std::shared_ptr<const NetDef>& net_def,
      Workspace* ws);

 protected:
  bool Run() override;

  using SimpleNet::operators_;

 private:
  // The list of blobs to delete when each operator finishes its run.
  // This will be populated during construction time.
  vector<vector<Blob*>> delete_list_;

  C10_DISABLE_COPY_AND_ASSIGN(SimpleRefCountNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_SIMPLE_REFCOUNT_H_

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>

#include "Types.hpp"

namespace c10d {

// ProcessGroup is a base class that captures collective and point to
// point communication in a fixed set of processes.
//
// The functions specified in the class below describe the API alone;
// implementations are provided in subclasses.
//
// Every function that performs I/O is executed asynchronously by a
// thread pool owned by the ProcessGroup (by default). They return an
// object that can be used to wait for completion or error.
//
// The ProcessGroup can instantiate subgroups with fewer or an equal
// number of members. Implementations must take care that multiple
// process groups can be used in parallel and synchronize accordingly.
//
// The ProcessGroup assumes a fixed set of processes. If the set
// changes, existing instances must be destructed and instantiation
// and initialization must start from scratch. For members of the
// process group to find each other (referred to as rendezvous from
// hereon)
//
// Note on usage with CUDA tensors:
//
// Operations on CUDA tensors are assumed to be executed
// asynchronously. Therefore they may have not yet executed when the
// tensors are passed to a collective function. The collective
// functions themselves should not block, so access to these tensors
// must be done asynchronously, as well as signaling completion of the
// collective function. We want to enable the following pattern:
//
//   z = at::sum(x, y);
//   work = pg.allreduce(z)
//   // Do something with z
//
// To do so, we execute the work associated with the collective
// function on a separate CUDA stream. Upon completing, this stream
// notifies the stream that produced the tensor in z
// (cudaEventRecord). Before returning from the collective function,
// it adds an asychronous wait for the internal stream
// (cudaEventSynchronize). This way we retain the ability to write
// sequential code that executes asynchronously, without requiring the
// caller to perform explicit synchronization.
//
class ProcessGroup {
 public:
  class Work {
   public:
    virtual ~Work();

    // Checks if request has completed. Non-blocking operation.
    virtual bool isCompleted() const = 0;

    // Waits until request completes. Blocking operation.
    // Returns false if the work completed with an exception.
    virtual bool wait() = 0;

    // Returns exception if wait() returned false.
    virtual const std::exception& exception() const = 0;
  };

  explicit ProcessGroup(int rank, int size);
  virtual ~ProcessGroup();

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  virtual std::shared_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) = 0;

  virtual std::shared_ptr<Work> allreduce(
      std::vector<at::Tensor>& data,
      const AllreduceOptions& opts = AllreduceOptions()) = 0;

 protected:
  const int rank_;
  const int size_;
};

} // namespace c10d

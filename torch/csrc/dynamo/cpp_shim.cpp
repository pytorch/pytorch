#include <cuda_runtime.h>
#include <vector>
#include <utility>

#include <torch/csrc/dynamo/cpp_shim.h>

#include <ATen/record_function.h>

// TODO: thread safety
class CUDATimingQueue {
private:
  struct EventPair {
    cudaEvent_t start;
    cudaEvent_t end;

    EventPair() {
      cudaEventCreate(&start);
      cudaEventCreate(&end);
    }

    ~EventPair() {
      cudaEventDestroy(start);
      cudaEventDestroy(end);
    }
  };

  // Remember: indices point between elements.
  // https://blog.nelhage.com/2015/08/indices-point-between-elements/
  //
  //     -- len --
  // O O X X X X X P O
  //    ^- front  ^- back
  //
  // NB: we don't advance end pointer until recordEnd, so end is a valid
  // pointer to the in progress cuda event pair

  std::vector<EventPair> event_pool_;
  // The queue is defined as the oldest events being in FRONT of the queue,
  // and the new events we're adding in the BACK of the queue.
  // Perhaps counter-intuitively, the back of the queue is a larger index if
  // we haven't wrapped around.
  // NB: do not test front_ == back_, you cannot distinguish full/empty.
  size_t front_ = 0; // event_pool[front] is the next event to check for completion, UNLESS len == 0
  size_t back_ = 0; // event_pool[back] is the next free element, UNLESS len == event_pool.size()
  size_t len_ = 0;
  size_t nesting_ = 0; // TODO: TLS

public:
  CUDATimingQueue(size_t pool_size = 10) {
    event_pool_.resize(pool_size);
  }

  ~CUDATimingQueue() = default;

  void recordStart() {
    nesting_++;
    if (nesting_ != 1) {
      return;
    }
    // Put the new event in the back of the queue
    TORCH_INTERNAL_ASSERT(len_ != event_pool_.size());
    cudaEventRecord(event_pool_[back_].start);
  }

  void recordEnd() {
    nesting_--;
    if (nesting_ != 0) {
      return;
    }
    cudaEventRecord(event_pool_[back_].end);
    // NOW advance the pointer
    back_ = (back_ + 1) % event_pool_.size();
    len_++;

    while (len_ > 0) {
      // Try to complete events
      cudaError_t status = cudaEventQuery(event_pool_[front_].end);
      if (status != cudaSuccess) break;
      float milliseconds;
      cudaEventElapsedTime(&milliseconds,
                 event_pool_[front_].start,
                 event_pool_[front_].end);
      std::cerr << "timing: " << milliseconds << "ms\n";
      front_ = (front_ + 1) % event_pool_.size();
      len_--;
    }
  }
};

struct _PytorchRecordFunctionState {
  at::RecordFunction guard;

  _PytorchRecordFunctionState() : guard(at::RecordScope::FUNCTION) {}
};

// Leak it!
static CUDATimingQueue* queue = nullptr;

void _compiled_region_enter() {
  // TODO: thread safety
  if (queue == nullptr) {
    queue = new CUDATimingQueue();
  }
  queue->recordStart();
}

void _compiled_region_exit() {
  TORCH_INTERNAL_ASSERT(queue);
  queue->recordEnd();
}

_PytorchRecordFunctionState* _pytorch_record_function_enter(const char* name) {
  _PytorchRecordFunctionState* state = new _PytorchRecordFunctionState();
  state->guard.before(name);
  return state;
}

void _pytorch_record_function_exit(_PytorchRecordFunctionState* state) {
  if (state == nullptr) {
    return;
  }
  delete state;
}

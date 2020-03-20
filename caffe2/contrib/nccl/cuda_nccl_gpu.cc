#include "caffe2/contrib/nccl/cuda_nccl_gpu.h"

namespace caffe2 {
namespace nccl {
namespace {

std::vector<int> getDevices(const NCCLExecution& ex) {
  std::vector<int> result;
  result.reserve(ex.elements.size());
  for (const auto& el : ex.elements) {
    result.push_back(el.device);
  }
  return result;
}

class NCCLContext {
 public:
  explicit NCCLContext(const NCCLExecution& ex)
      : devices_(getDevices(ex)), master_gpu_id_(ex.stream_gpu_id) {
    comms_.resize(devices_.size());
    CAFFE_NCCL_CHECK(
        ncclCommInitAll(comms_.data(), devices_.size(), devices_.data()));

    streams_.resize(devices_.size());
    events_.resize(devices_.size());
    for (auto i = 0U; i < devices_.size(); ++i) {
      CUDAGuard g(devices_[i]);
      // get stream priorities
      int lo_pri, hi_pri;
      CUDA_ENFORCE(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
#ifndef __HIP_PLATFORM_HCC__
      CUDA_ENFORCE(cudaStreamCreateWithPriority(
          &streams_[i], cudaStreamNonBlocking, hi_pri));
#else
      CUDA_ENFORCE(cudaStreamCreateWithFlags(
          &streams_[i], cudaStreamNonBlocking));
#endif // __HIP_PLATFORM_HCC__
      CUDA_ENFORCE(cudaEventCreateWithFlags(
          &events_[i], cudaEventDefault | cudaEventDisableTiming));
    }
    CUDAGuard g(master_gpu_id_);
    CUDA_ENFORCE(cudaEventCreateWithFlags(
        &master_event_, cudaEventDefault | cudaEventDisableTiming));
  }

  ~NCCLContext() {
    for (auto i = 0U; i < devices_.size(); ++i) {
      CUDAGuard g(devices_[i]);
      CUDA_ENFORCE(cudaStreamDestroy(streams_[i]));
      CUDA_ENFORCE(cudaEventDestroy(events_[i]));
    }
    CUDAGuard g(master_gpu_id_);
    CUDA_ENFORCE(cudaEventDestroy(master_event_));

    for (auto& comm : comms_) {
      ncclCommDestroy(comm);
    }
  }

  std::vector<int> devices_;
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
  int master_gpu_id_;
  cudaEvent_t master_event_;
  std::vector<cudaEvent_t> events_;

  C10_DISABLE_COPY_AND_ASSIGN(NCCLContext);
};

// We share the contexts across multiple operators, hence the cache.
static std::mutex& gContextsMutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::unique_ptr<NCCLContext>>& gContexts() {
  static std::unordered_map<std::string, std::unique_ptr<NCCLContext>> m;
  return m;
}

std::string ncclKey(const NCCLExecution& ex) {
  std::string result;
  int curr_device;
  CUDA_CHECK(cudaGetDevice(&curr_device));
  result += to_string(curr_device) + ":";
  for (const auto& el : ex.elements) {
    result += to_string(el.device) + ",";
  }
  return result;
}

NCCLContext* getNCCLContext(const NCCLExecution& ex) {
  auto& contexts = gContexts();
  const auto key = ncclKey(ex);
  if (!contexts[key]) {
    LOG(INFO) << "Creating NCCLContext for key: " << key;
    contexts[key].reset(new NCCLContext(ex));
  }
  return CHECK_NOTNULL(contexts[key].get());
}

template <typename T>
class ncclTypeWrapper;

template <>
class ncclTypeWrapper<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};

template <>
class ncclTypeWrapper<int> {
 public:
  static const ncclDataType_t type = ncclInt;
};

#ifdef CAFFE_HAS_CUDA_FP16
template <>
class ncclTypeWrapper<at::Half> {
 public:
  static const ncclDataType_t type = ncclHalf;
};
#endif

template <typename T, typename InitF, typename F>
void runNCCL(const NCCLExecution& ex, InitF&& init_f, F&& f) {
  // do initialization
  for (auto i = 0U; i < ex.elements.size(); ++i) {
    auto& ctx = ex.elements[i];
    CUDAGuard g(ctx.device);
    init_f(ex.elements[i]);
  }

  std::lock_guard<std::mutex> g(gContextsMutex());
  auto* context = getNCCLContext(ex);
  auto& comms = context->comms_;
  auto& streams = context->streams_;
  auto& events = context->events_;
  // Record an event on the master context, wait on it in each of the
  // children streams, so the children streams are synchronized WRT
  // the original stream.
  {
    CUDAGuard g(ex.stream_gpu_id);
    CUDA_ENFORCE(cudaEventRecord(context->master_event_, ex.stream));
  }

  {
    // lock out alloc / free while NCCL launches
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());

#if NCCL_VERSION_MIN(2, 0, 0)
    CAFFE_NCCL_CHECK(ncclGroupStart());
#endif

    for (auto i = 0U; i < ex.elements.size(); ++i) {
      auto& ctx = ex.elements[i];
      CUDAGuard g(ctx.device);
      auto& comm = comms[i];
      auto& stream = streams[i];

      DCHECK_EQ(ctx.device, GetGPUIDForPointer(ctx.src->raw_data()));
      CUDA_ENFORCE(cudaStreamWaitEvent(stream, context->master_event_, 0));
      f(ctx, comm, stream);
    }

#if NCCL_VERSION_MIN(2, 0, 0)
    CAFFE_NCCL_CHECK(ncclGroupEnd());
#endif

    for (auto i = 0U; i < ex.elements.size(); ++i) {
      auto& ctx = ex.elements[i];
      CUDAGuard g(ctx.device);
      auto& stream = streams[i];
      auto& event = events[i];

      // Record an event on each children stream that we have finished
      // our computation
      CUDA_ENFORCE(cudaEventRecord(event, stream));
    }
  }

  // Now, wait on all the events in the original stream.
  CUDAGuard dg(ex.stream_gpu_id);
  for (auto& event : events) {
    CUDA_ENFORCE(cudaStreamWaitEvent(CHECK_NOTNULL(ex.stream), event, 0));
  }
}

} // namespace

void destroyContexts() {
  std::lock_guard<std::mutex> g(gContextsMutex());
  auto& contexts = gContexts();
  contexts.clear();
}

template <typename T>
void NCCL<T>::AllReduce(const NCCLExecution& ex) {
  return runNCCL<T>(
      ex,
      [](const NCCLElement& ctx) {
        ctx.dst->Resize(ctx.src->sizes());
        ctx.dst->template mutable_data<T>();
      },
      [](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        CAFFE_NCCL_CHECK(ncclAllReduce(
            ctx.src->raw_data(),
            ctx.dst->raw_mutable_data(),
            ctx.dst->numel(),
            ncclTypeWrapper<T>::type,
            ncclSum,
            comm,
            stream));
      });
}

template <typename T>
void NCCL<T>::Broadcast(const NCCLExecution& ex) {
  return runNCCL<T>(
      ex,
      [](const NCCLElement& ctx) {
        ctx.dst->Resize(ctx.src->sizes());
        ctx.dst->template mutable_data<T>();
      },
      [&ex](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        CAFFE_NCCL_CHECK(ncclBcast(
            ctx.dst->raw_mutable_data(),
            ctx.dst->numel(),
            ncclTypeWrapper<T>::type,
            ex.root,
            comm,
            stream));
      });
}

template <typename T>
void NCCL<T>::Reduce(const NCCLExecution& ex) {
  return runNCCL<T>(
      ex,
      [](const NCCLElement& ctx) {
        if (ctx.dst) {
          ctx.dst->Resize(ctx.src->sizes());
          ctx.dst->template mutable_data<T>();
        }
      },
      [&ex](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        CAFFE_NCCL_CHECK(ncclReduce(
            ctx.src->raw_data(),
            ctx.dst ? ctx.dst->raw_mutable_data() : nullptr,
            ctx.src->numel(),
            ncclTypeWrapper<T>::type,
            ncclSum,
            ex.root,
            comm,
            stream));
      });
}

template <typename T>
void NCCL<T>::AllGather(const NCCLExecution& ex) {
  const auto n = ex.elements.size();
  return runNCCL<T>(
      ex,
      [n](const NCCLElement& ctx) {
        CAFFE_ENFORCE_NE(ctx.src, ctx.dst);
        std::vector<int64_t> dims;
        dims.reserve(ctx.src->dim() + 1);
        dims.push_back(n);
        for (auto d : ctx.src->sizes()) {
          dims.push_back(d);
        }
        ctx.dst->Resize(dims);
        ctx.dst->template mutable_data<T>();
      },
      [](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
#if NCCL_VERSION_MIN(2, 0, 0)
        CAFFE_NCCL_CHECK(ncclAllGather(
            ctx.src->raw_data(),
            ctx.dst->raw_mutable_data(),
            ctx.src->numel(),
            ncclTypeWrapper<T>::type,
            comm,
            stream));
#else
        CAFFE_NCCL_CHECK(ncclAllGather(
            ctx.src->raw_data(),
            ctx.src->size(),
            ncclTypeWrapper<T>::type,
            ctx.dst->raw_mutable_data(),
            comm,
            stream));
#endif
      });
}

template <typename T>
void NCCL<T>::ReduceScatter(const NCCLExecution& ex) {
  return runNCCL<T>(
      ex,
      [](const NCCLElement& ctx) {
        CAFFE_ENFORCE_NE(ctx.src, ctx.dst);
        const auto& srcDims = ctx.src->sizes();
        std::vector<int64_t> dstDims(srcDims.begin() + 1, srcDims.end());
        ctx.dst->Resize(dstDims);
        ctx.dst->template mutable_data<T>();
      },
      [](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        CAFFE_NCCL_CHECK(ncclReduceScatter(
            ctx.src->raw_data(),
            ctx.dst->raw_mutable_data(),
            ctx.dst->numel(),
            ncclTypeWrapper<T>::type,
            ncclSum,
            comm,
            stream));
      });
}

// Explicit instantiation
template class NCCL<float>;
template class NCCL<int>;
#ifdef CAFFE_HAS_CUDA_FP16
template class NCCL<at::Half>;
#endif

} // namespace nccl
} // namespace caffe2

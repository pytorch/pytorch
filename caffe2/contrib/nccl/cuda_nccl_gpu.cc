#include "cuda_nccl_gpu.h"

namespace caffe2 {

namespace nccl {

namespace {

// We share the contexts across multiple operators, hence the
// thread-local cache
static std::mutex& gContextsMutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::unique_ptr<NCCLContext>>& gContexts() {
  // Initiazed after CUDA, so guaranteed to be destructed before CUDA.
  static std::unordered_map<std::string, std::unique_ptr<NCCLContext>> m;
  return m;
}

std::string ncclKey(const std::vector<NCCLElement>& ctxs) {
  std::string result;
  for (const auto& ctx : ctxs) {
    result += std::to_string(ctx.device) + ",";
  }
  return result;
}

NCCLContext* getNCCLContexts(const std::vector<NCCLElement>& ctxs) {
  auto& contexts = gContexts();
  const auto key = ncclKey(ctxs);
  if (!contexts[key]) {
    LOG(INFO) << "Creating NCCLContext for key: " << key;
    std::vector<int> devices;
    devices.reserve(ctxs.size());
    for (const auto& ctx : ctxs) {
      devices.push_back(ctx.device);
    }
    contexts[key].reset(new NCCLContext(devices));
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

#ifdef CUDA_HAS_HALF
template <>
class ncclTypeWrapper<float16> {
 public:
  static const ncclDataType_t type = ncclHalf;
};
#endif
}

template<typename T, typename F>
void runNCCL(const std::vector<NCCLElement>& ctxs, F&& f) {
  std::lock_guard<std::mutex> g(gContextsMutex());
  auto* context = getNCCLContexts(ctxs);
  auto& comms = context -> comms();
  auto& streams = context -> streams();

  for (auto i = 0; i < ctxs.size(); ++i) {
    auto& ctx = ctxs[i];
    DeviceGuard g(ctx.device);
    auto& comm = comms[i];
    auto& stream = streams[i];
    CHECK_EQ(ctx.device, GetGPUIDForPointer(ctx.src->raw_data()));
    f(ctx, comm, stream);
  }
  for (auto i = 0; i < ctxs.size(); ++i) {
    DeviceGuard g(ctxs[i].device);
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
}

template <typename T>
void NCCL<T>::AllReduce(const std::vector<NCCLElement>& ctxs) {
  return runNCCL<T>(
      ctxs,
      [](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        ctx.dst->Reshape(ctx.src->dims());
        ctx.dst->template mutable_data<T>();
        CAFFE_NCCL_CHECK(ncclAllReduce(
            ctx.src->raw_data(),
            ctx.dst->raw_mutable_data(),
            ctx.dst->size(),
            ncclTypeWrapper<T>::type,
            ncclSum,
            comm,
            stream));
      });
}

template <typename T>
void NCCL<T>::Broadcast(const std::vector<NCCLElement>& ctxs, int root) {
  return runNCCL<T>(
      ctxs,
      [root](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        ctx.dst->Reshape(ctx.src->dims());
        ctx.dst->template mutable_data<T>();
        CAFFE_NCCL_CHECK(ncclBcast(
            ctx.dst->raw_mutable_data(),
            ctx.dst->size(),
            ncclTypeWrapper<T>::type,
            root,
            comm,
            stream));
      });
}

template <typename T>
void NCCL<T>::Reduce(const std::vector<NCCLElement>& ctxs, int root) {
  return runNCCL<T>(
      ctxs,
      [root](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        if (ctx.dst) {
          ctx.dst->Reshape(ctx.src->dims());
          ctx.dst->template mutable_data<T>();
        }
        CAFFE_NCCL_CHECK(ncclReduce(
            ctx.src->raw_data(),
            ctx.dst ? ctx.dst->raw_mutable_data() : nullptr,
            ctx.src->size(),
            ncclTypeWrapper<T>::type,
            ncclSum,
            root,
            comm,
            stream));
      });
}

template <typename T>
void NCCL<T>::AllGather(const std::vector<NCCLElement>& ctxs) {
  const auto n = ctxs.size();
  return runNCCL<T>(
      ctxs,
      [n](const NCCLElement& ctx, ncclComm_t comm, cudaStream_t stream) {
        CHECK_NE(ctx.src, ctx.dst);
        std::vector<TIndex> dims;
        dims.reserve(ctx.src->ndim() + 1);
        dims.push_back(n);
        for (auto d: ctx.src->dims()) {
          dims.push_back(d);
        }
        ctx.dst->Reshape(dims);
        ctx.dst->template mutable_data<T>();
        CAFFE_NCCL_CHECK(ncclAllGather(
            ctx.src->raw_data(),
            ctx.src->size(),
            ncclTypeWrapper<T>::type,
            ctx.dst->raw_mutable_data(),
            comm,
            stream));
      });
}

// Explicit instantiation
template class NCCL<float>;
#ifdef CUDA_HAS_HALF
template class NCCL<float16>;
#endif
}
}

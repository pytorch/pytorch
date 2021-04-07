#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/slice_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {
__global__ void SliceCopyKernel(
    char* src_offset_bytes,
    int src_block_size_bytes,
    char* dst_offset_bytes,
    int dst_block_size_bytes,
    int copy_size,
    int itemsize,
    int num_blocks) {
  if ((copy_size % sizeof(int) == 0) &&
      (src_block_size_bytes % sizeof(int) == 0) &&
      (dst_block_size_bytes % sizeof(int) == 0)) {
    int* src = (int*)src_offset_bytes;
    int* dst = (int*)dst_offset_bytes;

    int src_block_size = src_block_size_bytes / sizeof(int);
    int dst_block_size = dst_block_size_bytes / sizeof(int);

    int copyChunks = copy_size / sizeof(int);

    CUDA_1D_KERNEL_LOOP(index, num_blocks * copyChunks) {
      int chunk = index % copyChunks;
      int block = index / copyChunks;

      dst[block * dst_block_size + chunk] = src[block * src_block_size + chunk];
    }
  } else {
    char* src = (char*)src_offset_bytes;
    char* dst = (char*)dst_offset_bytes;

    int src_block_size = src_block_size_bytes / sizeof(char);
    int dst_block_size = dst_block_size_bytes / sizeof(char);

    int copyChunks = copy_size / sizeof(char);

    CUDA_1D_KERNEL_LOOP(index, num_blocks * copyChunks) {
      int chunk = index % copyChunks;
      int block = index / copyChunks;

      dst[block * dst_block_size + chunk] = src[block * src_block_size + chunk];
    }
  }
}

template <class SIndex, class Context>
bool SliceImplGpu(
    Tensor* output,
    const Tensor& data,
    const TensorCPU& starts,
    const TensorCPU& ends,
    Context* context,
    Tensor* gdata = nullptr,
    const Tensor* go = nullptr) {
  bool backward = output == nullptr;

  auto* starts_data = starts.template data<SIndex>();
  auto* ends_data = ends.template data<SIndex>();

  CAFFE_ENFORCE_EQ(starts.dim(), 1);
  CAFFE_ENFORCE_EQ(ends.dim(), 1);
  CAFFE_ENFORCE_GE(data.dim(), starts.size());
  CAFFE_ENFORCE_EQ(starts.numel(), ends.numel());

  std::vector<int> starts_idx(data.dim());
  std::vector<int> ends_idx(data.dim());
  std::vector<int> dst_sizes(data.dim());

  for (int i = 0; i < data.dim(); ++i) {
    if (i >= starts.numel()) {
      starts_idx[i] = 0;
      ends_idx[i] = data.size(i);
      dst_sizes[i] = data.size(i);
      continue;
    }
    if (data.size(i) > 0) {
      auto start = starts_data[i];
      auto end = ends_data[i];
      if (start < 0) {
        start = data.size(i) + 1 + start;
      }
      if (end < 0) {
        end = data.size(i) + 1 + end;
      }
      if (start > data.size(i)) {
        start = data.size(i);
      }
      if (end > data.size(i)) {
        end = data.size(i);
      }
      CAFFE_ENFORCE_GE(start, 0);
      CAFFE_ENFORCE_GE(end, 0);
      CAFFE_ENFORCE_GE(end, start);
      starts_idx[i] = start;
      ends_idx[i] = end;
      dst_sizes[i] = end - start;
    } else {
      starts_idx[i] = 0;
      ends_idx[i] = 0;
      dst_sizes[i] = 0;
    }
  }

  if (data.numel() <= 0) {
    // When the input is empty, we do not need to do copy.
    if (!backward) {
      output->Resize(dst_sizes);
      output->raw_mutable_data(data.meta());
    }
    return true;
  }
  // for now only supports slicing in 1 dimension
  int dim = -1;
  for (int i = 0; i < data.dim(); ++i) {
    if (starts_idx[i] > 0 || ends_idx[i] < data.size(i)) {
      CAFFE_ENFORCE_EQ(
          dim, -1, "Currently only possible to slice in 1 dimension.");
      dim = i;
    }
  }
  if (dim == -1) {
    if (!backward) {
      output->CopyFrom(data, true /*async*/);
    } else {
      gdata->CopyFrom(*go, true /*async*/);
    }
    return true;
  }
  int unit = std::accumulate(
      data.sizes().begin() + dim + 1,
      data.sizes().end(),
      1,
      std::multiplies<int>());
  int num_blocks = std::accumulate(
      data.sizes().begin(),
      data.sizes().begin() + dim,
      1,
      std::multiplies<int>());
  if (!backward) {
    output->Resize(dst_sizes);
  } else {
    gdata->ResizeLike(data);
  }

  auto itemsize = data.meta().itemsize();

  if (!backward) {
    char* src_bytes = (char*)data.raw_data();
    char* dst_bytes = (char*)output->raw_mutable_data(data.meta());

    size_t src_nbytes = data.nbytes();
    size_t dst_nbytes = output->nbytes();

    size_t src_block_size = unit * data.size(dim);
    size_t dst_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
    size_t src_offset = unit * starts_idx[dim];

    if (num_blocks == 0 || dst_block_size == 0) {
      return true;
    }

    size_t src_block_size_bytes = itemsize * src_block_size;
    size_t dst_block_size_bytes = itemsize * dst_block_size;
    char* src_offset_bytes = src_bytes + itemsize * src_offset;
    char* dst_offset_bytes = dst_bytes;

    SliceCopyKernel<<<
        std::min(num_blocks, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(
        src_offset_bytes,
        src_block_size_bytes,
        dst_offset_bytes,
        dst_block_size_bytes,
        dst_block_size_bytes,
        itemsize,
        num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    char* src_bytes = (char*)go->raw_data();
    char* dst_bytes = (char*)gdata->raw_mutable_data(go->meta());

    size_t src_nbytes = go->nbytes();
    size_t dst_nbytes = gdata->nbytes();

    size_t src_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
    size_t dst_block_size = unit * data.size(dim);
    size_t dst_offset = unit * starts_idx[dim];

    if (num_blocks == 0 || dst_block_size == 0) {
      return true;
    }

    size_t src_block_size_bytes = itemsize * src_block_size;
    size_t dst_block_size_bytes = itemsize * dst_block_size;

    char* src_offset_bytes = src_bytes;
    char* dst_offset_bytes = dst_bytes + itemsize * dst_offset;
    // Zero out gradient blob before copy since we copy in fewer items than
    // there is space for
    math::Set<float, CUDAContext>(
        gdata->numel(),
        0.0f,
        (float*)gdata->raw_mutable_data(go->meta()),
        context);

    // If output tensor is empty, just return zeroed gradient tensor
    if (!src_bytes) {
      return true;
    }

    SliceCopyKernel<<<
        std::min(num_blocks, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(
        src_offset_bytes,
        src_block_size_bytes,
        dst_offset_bytes,
        dst_block_size_bytes,
        src_block_size_bytes,
        itemsize,
        num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

} // namespace

template<>
class SliceOp<CUDAContext> : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  template<class... Args> explicit SliceOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        starts_(this->template GetRepeatedArgument<int64_t>("starts")),
        ends_(this->template GetRepeatedArgument<int64_t>("ends")),
        statically_inited_(false) {}

  bool RunOnDevice() override {
    if (InputSize() > 1) {
      return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
    } else {
      return DoRunWithType<int64_t>();
    }
  }

  template <typename SIndex>
  bool DoRunWithType() {
    auto* output = Output(0);
    auto& data = Input(0);

    if (InputSize() > 1) {
      ReinitializeAndCopyFrom(&starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
      ReinitializeAndCopyFrom(&ends_host_, at::dtype<SIndex>().device(CPU), Input(2));
    } else {
      if (!statically_inited_) {
        CAFFE_ENFORCE(HasArgument("starts"));
        CAFFE_ENFORCE(HasArgument("ends"));
        CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

        ReinitializeTensor(&starts_host_, {static_cast<int64_t>(starts_.size())}, at::dtype<SIndex>().device(CPU));
        ReinitializeTensor(&ends_host_, {static_cast<int64_t>(ends_.size())}, at::dtype<SIndex>().device(CPU));

        memcpy(
            starts_host_.mutable_data<SIndex>(),
            starts_.data(),
            sizeof(SIndex) * starts_.size());
        memcpy(
            ends_host_.mutable_data<SIndex>(),
            ends_.data(),
            sizeof(SIndex) * ends_.size());
        statically_inited_ = true;
      }
    }

    return SliceImplGpu<SIndex, CUDAContext>(
        output, data, starts_host_, ends_host_, &context_);
  }
 private:
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  bool statically_inited_;
  Tensor starts_host_;
  Tensor ends_host_;

};  // class SliceOp<CUDAContext>

REGISTER_CUDA_OPERATOR(Slice, SliceOp<CUDAContext>);

template <>
class SliceGradientOp<CUDAContext> : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  template<class... Args> explicit SliceGradientOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        starts_(this->template GetRepeatedArgument<int64_t>("starts")),
        ends_(this->template GetRepeatedArgument<int64_t>("ends")),
        statically_inited_(false) {}

  C10_DISABLE_COPY_AND_ASSIGN(SliceGradientOp);

  bool RunOnDevice() override {
    if (InputSize() == 4) {
      return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
    } else {
      return DoRunWithType<int64_t>();
    }
  }

  template <typename SIndex>
  bool DoRunWithType() {
    auto* gdata = Output(0);
    auto& data = Input(0);

    if (InputSize() == 4) {
      ReinitializeAndCopyFrom(&starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
      ReinitializeAndCopyFrom(&ends_host_, at::dtype<SIndex>().device(CPU), Input(2));

      auto& go = Input(3);

      return SliceImplGpu<SIndex, CUDAContext>(
          nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
    } else {
      if (!statically_inited_) {
        CAFFE_ENFORCE(HasArgument("starts"));
        CAFFE_ENFORCE(HasArgument("ends"));
        CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

        ReinitializeTensor(&starts_host_, {static_cast<int64_t>(starts_.size())}, at::dtype<SIndex>().device(CPU));
        ReinitializeTensor(&ends_host_, {static_cast<int64_t>(ends_.size())}, at::dtype<SIndex>().device(CPU));

        memcpy(
            starts_host_.mutable_data<SIndex>(),
            starts_.data(),
            sizeof(SIndex) * starts_.size());
        memcpy(
            ends_host_.mutable_data<SIndex>(),
            ends_.data(),
            sizeof(SIndex) * ends_.size());

        statically_inited_ = true;
      }
      auto& go = Input(1);

      return SliceImplGpu<SIndex, CUDAContext>(
          nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
    }
  }
 private:

  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  bool statically_inited_;
  Tensor starts_host_;
  Tensor ends_host_;
};  // class SliceGradientOp<CUDAContext>
REGISTER_CUDA_OPERATOR(SliceGradient, SliceGradientOp<CUDAContext>);
} // namespace caffe2

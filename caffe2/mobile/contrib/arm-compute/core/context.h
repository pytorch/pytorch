#ifndef CAFFE2_OPENGL_CONTEXT_H_
#define CAFFE2_OPENGL_CONTEXT_H_

#ifdef CAFFE2_OPENGL_BACKEND
#error Can only build one OpenGL backend at a time.
#else
#define CAFFE2_OPENGL_BACKEND
#endif

#include "caffe2/core/allocator.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCFunctions.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"
#include "include/half/half.hpp"

namespace caffe2 {

typedef half_float::half half;
typedef half DataType;

template <typename T> class GLTensor;

class GLContext final {
public:
  static bool initialized;
  explicit GLContext();
  explicit GLContext(const DeviceOption &option) {
    DCHECK_EQ(option.device_type(), OPENGL);
    GLContext();
  }
  ~GLContext() {}

  static void sync() { arm_compute::GCScheduler::get().memory_barrier(); }

  template <typename T>
  using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

  template <typename T>
  static deleted_unique_ptr<const GLTensor<T>> getGLTensor(const Blob *b) {
    if (b->IsType<TensorCPU>()) {
      auto &Xcpu = b->Get<TensorCPU>();
      GLTensor<T> *X_raw_ptr;
      X_raw_ptr = new GLTensor<T>();
      X_raw_ptr->ResizeLike(Xcpu);
      deleted_unique_ptr<const GLTensor<T>> X_unique_ptr(
          X_raw_ptr, [](const GLTensor<T> *X) { delete X; });
      return X_unique_ptr;
    }
    const GLTensor<T> *X_raw_ptr;
    X_raw_ptr = &b->Get<GLTensor<T>>();
    deleted_unique_ptr<const GLTensor<T>> X_unique_ptr(
        X_raw_ptr, [](const GLTensor<T> *X) { return; });
    return X_unique_ptr;
  }

  /*
   * Everything below is basically boiler plate for Context classes
   */
  static std::pair<void *, MemoryDeleter> New(size_t nbytes) {
    return std::pair<void *, MemoryDeleter>(malloc(nbytes), GLContext::Delete);
  }

  static void Delete(void *data) {
    if (data != nullptr) {
      free(data);
    }
  }

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void *src, void *dst) {}

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T *src, T *dst) {
    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                      static_cast<const void *>(src),
                                      static_cast<void *>(dst));
  }

  template <class SrcContext, class DstContext>
  inline void CopyItems(const TypeMeta &meta, size_t n, const void *src,
                        void *dst) {
    CAFFE_ENFORCE(!meta.copy(), "GLContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  void SwitchToDevice(int a, ...) { /* TODO */
  }
  void SwitchToDevice() { SwitchToDevice(0); }

  inline void WaitEvent(const Event &ev) { /* TODO */
  }
  void FinishDeviceComputation() { /* TODO */
  }
  inline void Record(Event *ev, const char *&) const { /* TODO */
  }
  static bool IsStreamFree(const DeviceOption& /* unused */, int /* unused */) {
    return true;
  }
  bool HasAsyncPartDefault() const { return false; }
  bool SupportsAsyncScheduling() const { return false; }

};

template <typename T> class GLTensor {
private:
  bool allocated_ = false;
public:
  GLTensor() { tensor_ = make_unique<arm_compute::GCTensor>(); }
  ~GLTensor() { tensor_->allocator()->free(); }

  template <typename TensorType> void ResizeLike(TensorType &X) {
    tensor_->allocator()->free();
    SetDims(X.dims());
    shape_ = arm_compute::TensorShape();
    for (int i = 0; i < dims_.size(); i++) {
      shape_.set(dims_.size() - i - 1, dims_[i]);
    }

    tensor_->allocator()->init(
        arm_compute::TensorInfo(shape_, 1, arm_compute::DataType::F16));
  }

  template <typename... Ts> void Resize(Ts... dim_source) {
    bool size_changed = SetDims(dim_source...);
    if (size_changed) {
      // TODO: Make it type generic
      int64_t new_size = size_ * sizeof(T);
      tensor_->allocator()->free();
      for (int i = 0; i < dims_.size(); i++) {
        shape_.set(dims_.size() - i - 1, dims_[i]);
      }
      tensor_->allocator()->init(
          arm_compute::TensorInfo(shape_, 1, arm_compute::DataType::F16));
    }
  }

  // Allocates and copies data if needed
  void lazy_allocate(const Blob *b, bool allocate_tensor, bool try_to_copy_from_cpu) const {
    if (try_to_copy_from_cpu) {
      // we skip GLTensors, nothing to copy
      if (!b->IsType<GLTensor>()) {
        // typically only called on the second run
        if (allocate_tensor) {
          allocate();
        }
        fillGLTensor(b);
      }
    }
  }

  void allocate() const {
    tensor_->allocator()->allocate();
  }

  void fillGLTensor(const Blob *b) const {
    if (b->IsType<TensorCPU>()) {
      auto &Xcpu = b->Get<TensorCPU>();

      T *buffer = map();
      char *byte_buffer = (char *)buffer;
      auto info = tensor_->info();
      if (Xcpu.ndim() == 4) {
        auto M = Xcpu.dim32(0);
        auto C = Xcpu.dim32(1);
        auto H = Xcpu.dim32(2);
        auto W = Xcpu.dim32(3);
        for (auto m = 0; m < M; ++m) {
          for (auto c = 0; c < C; ++c) {
            for (auto h = 0; h < H; ++h) {
              for (auto w = 0; w < W; ++w) {
                T *b = (T *)(&byte_buffer[info->offset_element_in_bytes(
                    arm_compute::Coordinates(w, h, c, m))]);
                // require cpu input blob to be float
                *b = T(Xcpu.data<float>()[((m * C + c) * H + h) * W + w]);
              }
            }
          }
        }
      } else if (Xcpu.ndim() == 3) {
        auto C = Xcpu.dim32(0);
        auto H = Xcpu.dim32(1);
        auto W = Xcpu.dim32(2);
        for (auto c = 0; c < C; ++c) {
          for (auto h = 0; h < H; ++h) {
            for (auto w = 0; w < W; ++w) {
              T *b = (T *)(&byte_buffer[info->offset_element_in_bytes(
                  arm_compute::Coordinates(w, h, c))]);
              // require cpu input blob to be float
              *b = T(Xcpu.data<float>()[(c * H + h) * W + w]);
            }
          }
        }
      } else if (Xcpu.ndim() == 2) {
        auto H = Xcpu.dim32(0);
        auto W = Xcpu.dim32(1);
        for (auto h = 0; h < H; ++h) {
          for (auto w = 0; w < W; ++w) {
            T *b = (T *)(&byte_buffer[info->offset_element_in_bytes(
                arm_compute::Coordinates(w, h))]);
            // require cpu input blob to be float
            *b = T(Xcpu.data<float>()[h * W + w]);
          }
        }
      } else {
        auto size = Xcpu.dim32(0);
        for (auto i = 0; i < size; ++i) {
          T *b = (T *)(&byte_buffer[info->offset_element_in_bytes(arm_compute::Coordinates(i))]);
          // require cpu input blob to be float
          *b = T(Xcpu.data<float>()[i]);
        }
      }
      unmap();
    }
  }


  const int32_t ndim() const { return dims_.size(); }

  vector<TIndex> dims() const { return dims_; }

  const int32_t dim32(const int index) const { return dims_.at(index); }

  const int32_t size() const {
    int32_t s = 1;
    for (int i = 0; i < dims_.size(); i++) {
      s *= dims_[i];
    }
    return s;
  }

  arm_compute::GCTensor *get_underlying() const { return tensor_.get(); }

  T *map() const {
    GLContext::sync();
    tensor_->map(true);
    return reinterpret_cast<T *>(tensor_->buffer());
  }

  void unmap() const { return tensor_->unmap(); }

  void sync() const {
    GLContext::sync();
    tensor_->map();
    tensor_->unmap();
  }

private:
  template <typename TI, typename = typename std::enable_if<
                             std::is_integral<TI>::value>::type>
  bool SetDims(const vector<TI> &src) {
    auto old_size = size_;
    dims_.resize(src.size());
    TIndex new_size = 1;
    for (unsigned int i = 0; i < src.size(); ++i) {
      new_size *= src[i];
      dims_[i] = src[i];
    }
    size_ = new_size;
    return size_ != old_size;
  }

  bool SetDims() {
    auto old_size = size_;
    dims_.resize(0);
    size_ = 1;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0) {
    auto old_size = size_;
    dims_.resize(1);
    dims_[0] = d0;
    size_ = d0;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1) {
    auto old_size = size_;
    dims_.resize(2);
    dims_[0] = d0;
    dims_[1] = d1;
    size_ = d0 * d1;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1, const TIndex d2) {
    auto old_size = size_;
    dims_.resize(3);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    size_ = d0 * d1 * d2;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1, const TIndex d2,
               const TIndex d3) {
    auto old_size = size_;
    dims_.resize(4);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    size_ = d0 * d1 * d2 * d3;
    return size_ != old_size;
  }

  vector<TIndex> dims_;
  TIndex size_ = -1;
  arm_compute::TensorShape shape_;
  unique_ptr<arm_compute::GCTensor> tensor_;
};

template<typename T = half>
void getTensorCPU(const GLTensor<T>& g_, TensorCPU& g) {
  g.Resize(g_.dims());
  T *buffer = g_.map();

  for (auto i = 0; i < g.size(); ++i) {
    auto tmp = buffer[i];
    g.mutable_data<float>()[i] = tmp;
  }
  g_.unmap();
}


} // namespace caffe2

#endif /* CAFFE2_OPENGL_CONTEXT_H_ */

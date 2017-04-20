#ifdef __ARM_NEON__

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "snpe_ffi.h"
#include <dlfcn.h>

namespace caffe2 {

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

class SNPEOp final : public Operator<CPUContext> {
 public:
  SNPEOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
        model_buffer_(OperatorBase::GetSingleArgument<string>("model_buffer", ""))
  {
    handle_ = deleted_unique_ptr<void>(dlopen("libsnpe_jni.so", RTLD_LAZY), [](void* handle) {
      if (handle) {
        dlclose(handle);
      }
    });
    if (!handle_.get()) {
      std::cerr << dlerror() << std::endl;
    }

    OPERATOR_NEEDS_FEATURE(handle_.get(), "Couldn't find libsnpe_jni.so");

#define X(n)                                    \
  dlerror();                                    \
  auto* n##_f = (typeof(&n))dlsym(handle_.get(), #n); \
  OPERATOR_NEEDS_FEATURE(n##_f, dlerror());

    {
      X(snpe_has_gpu);
      X(snpe_create);
      X(snpe_destroy);
      X(snpe_get_input_dims);
      X(snpe_run);
      X(snpe_copy_output_to);
    }

    X(snpe_has_gpu);
    OPERATOR_NEEDS_FEATURE(snpe_has_gpu_f(), "No GPU found, cannot use SNPE.");

    X(snpe_create)
#undef X

// Redefine to use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.

#define X(n)                                              \
      dlerror();                                          \
      auto* n##_f = (typeof(&n))dlsym(handle_.get(), #n); \
      CAFFE_ENFORCE(n##_f, dlerror());

    ctx_ = deleted_unique_ptr<void>(snpe_create_f(reinterpret_cast<const unsigned char *>(model_buffer_.data()),
          model_buffer_.length()), [this](void* ctx) {
      if (ctx) {
        X(snpe_destroy);
        snpe_destroy_f(ctx);
      }
    });
  }

  bool RunOnDevice() override {
    // TODO: fill in input/output.
    X(snpe_get_input_dims);
    size_t const* dims;
    size_t dimSize;
    snpe_get_input_dims_f(ctx_.get(), &dims, &dimSize);
    CAFFE_ENFORCE_EQ(Input(0).ndim(), dimSize);
    for (auto i = 0; i < dimSize; ++i) {
      CAFFE_ENFORCE_EQ(Input(0).dim(i), dims[i]);
    }

    X(snpe_run);
    snpe_run_f(ctx_.get(), Input(0).data<float>(), Input(0).size(), &dims, &dimSize);
    std::vector<int64_t> outputDims(dimSize);
    for (auto i = 0; i < dimSize; ++i) {
      outputDims[i] = dims[i];
    };

    Output(0)->Resize(outputDims);
    X(snpe_copy_output_to);
    snpe_copy_output_to_f(ctx_.get(), Output(0)->mutable_data<float>());
    CAFFE_ENFORCE(Output(0)->data<float>(), "nullptr where output should be!\n");
    return true;
  }

 private:
  string model_buffer_;
  deleted_unique_ptr<void> handle_;
  // needs to be destroyed *before* handle_
  deleted_unique_ptr<void> ctx_;
};

REGISTER_CPU_OPERATOR(SNPE, SNPEOp);
}

#undef X

#endif
